import numpy as np
from sklearn.cluster import KMeans

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    return covariances


def _estimate_gaussian_parameters(X, resp, reg_covar):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar)
    return nk, means, covariances

def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    means : array-like of shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision
    # matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

    if covariance_type == "full":
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "tied":
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "diag":
        precisions = precisions_chol ** 2
        log_prob = (
            np.sum((means ** 2 * precisions), 1)
            - 2.0 * np.dot(X, (means * precisions).T)
            + np.dot(X ** 2, precisions.T)
        )

    elif covariance_type == "spherical":
        precisions = precisions_chol ** 2
        log_prob = (
            np.sum(means ** 2, 1) * precisions
            - 2 * np.dot(X, means.T * precisions)
            + np.outer(row_norms(X, squared=True), precisions)
        )
    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

class GaussianMixture():
    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        self.n_components=n_components
        self.covariance_type = covariance_type
        self.max_iter=max_iter
        self.n_init=n_init
        self.weights_init = weights_init
        self.means_init = means_init
        self.reg_covar=reg_covar
        self.tol=tol
        self.precisions_init = precisions_init


    def fit(self, X, y=None):
        self.fit_predict(X, y)
        return self

    def fit_predict(self, X, y=None):
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_initial_parameters(X)
        n_init=self.n_init
        max_lower_bound = -np.inf

        for init in range(n_init):
            self._initialize_parameters(X)
            lower_bound = -np.inf
            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound
                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)

    def _e_step(self, X):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp


    def _check_initial_parameters(self, X):
        if self.n_components < 1:
            raise ValueError(
                "Invalid value for 'n_components': %d "
                "Estimation requires at least one component"
                % self.n_components
            )
        if self.max_iter < 1:
            raise ValueError(
                "Invalid value for 'max_iter': %d "
                "Estimation requires at least one iteration"
                % self.max_iter
            )
        if self.tol < 0.0:
            raise ValueError(
                "Invalid value for 'tol': %.5f "
                "Tolerance used by the EM must be non-negative"
                % self.tol
            )
        if self.n_init < 1:
            raise ValueError(
                "Invalid value for 'n_init': %d Estimation requires at least one run"
                % self.n_init
            )
        self._check_parameters(X)
    def _check_parameters(self, X):
        pass

    def _initialize_parameters(self, X):
        n_samples, _ = X.shape
        resp = np.zeros((n_samples, self.n_components))
        resp = np.zeros((n_samples, self.n_components))
        label = (
            KMeans(
                n_clusters=self.n_components, n_init=1
            )
            .fit(X)
            .labels_
        )
        resp[np.arange(n_samples), label] = 1
        self._initialize(X, resp)

    def _initialize(self, X, resp):
        n_samples, _ = X.shape

        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp ,self.reg_covar
        )
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

    def _estimate_log_prob_resp(self, X):
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        )

    def _estimate_log_weights(self):
        return np.log(self.weights_)









