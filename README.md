# ABQ
ABQ

# Expectation of the Sum of Absolute Coordinates on a Unit Sphere

## 1. Introduction

Let $\mathbf{q} \in \mathbb{R}^D$ be a random vector uniformly distributed on the unit sphere $S^{D-1}$. We are interested in the expected value of the sum of the absolute values of its coordinates, i.e., $\mathbb{E}\left[\sum_{i=1}^D |q_i|\right]$. This quantity arises naturally in the analysis of normalized vectors in high-dimensional spaces.

## 2. Preliminaries

Let $Z_1, Z_2, \dots, Z_D$ be independent and identically distributed (i.i.d.) standard normal random variables, i.e., $Z_i \sim \mathcal{N}(0,1)$. Then the random vector
\[
\mathbf{Z} = (Z_1, Z_2, \dots, Z_D)
\]
has a spherically symmetric distribution. Moreover, the normalized vector
\[
\mathbf{q} = \frac{\mathbf{Z}}{\|\mathbf{Z}\|}
\]
is uniformly distributed on $S^{D-1}$. Here, $\|\mathbf{Z}\| = \sqrt{\sum_{i=1}^D Z_i^2}$.

For a fixed coordinate $i$, define $U = q_i^2 = \frac{Z_i^2}{\sum_{j=1}^D Z_j^2}$. Then $U$ follows a Beta distribution:
\[
U \sim \operatorname{Beta}\left(\frac{1}{2}, \frac{D-1}{2}\right).
\]

## 3. Main Theorem

**Theorem 1.** Let $\mathbf{q}$ be uniformly distributed on the unit sphere $S^{D-1}$ in $\mathbb{R}^D$. Then the expected sum of the absolute coordinates is given by
\[
\mathbb{E}\left[\sum_{i=1}^D |q_i|\right] = \frac{D \,\Gamma\!\left(\frac{D}{2}\right)}{\sqrt{\pi} \;\Gamma\!\left(\frac{D+1}{2}\right)}.
\]

*Proof.* By the spherical symmetry, all coordinates are identically distributed. Therefore,
\[
\mathbb{E}\left[\sum_{i=1}^D |q_i|\right] = D \; \mathbb{E}[|q_1|].
\]
We now compute $\mathbb{E}[|q_1|]$. Using the normal representation, we have $|q_1| = \frac{|Z_1|}{\|\mathbf{Z}\|}$. Equivalently, $|q_1| = \sqrt{U}$ where $U \sim \operatorname{Beta}\left(\frac{1}{2}, \frac{D-1}{2}\right)$.

The $s$-th moment of a Beta-distributed random variable $U \sim \operatorname{Beta}(\alpha, \beta)$ is
\[
\mathbb{E}[U^s] = \frac{B(\alpha + s, \beta)}{B(\alpha, \beta)},
\]
where $B(\cdot, \cdot)$ is the Beta function. Setting $\alpha = \frac{1}{2}$, $\beta = \frac{D-1}{2}$, and $s = \frac{1}{2}$, we obtain
\[
\mathbb{E}[U^{1/2}] = \frac{B\left(1, \frac{D-1}{2}\right)}{B\left(\frac{1}{2}, \frac{D-1}{2}\right)}.
\]
Using the relationship $B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$ and the facts $\Gamma(1)=1$, $\Gamma(\frac{1}{2})=\sqrt{\pi}$, we simplify:
\[
\mathbb{E}[U^{1/2}] = \frac{\Gamma(1) \Gamma\!\left(\frac{D-1}{2}\right) / \Gamma\!\left(\frac{D+1}{2}\right)}{\Gamma\!\left(\frac{1}{2}\right) \Gamma\!\left(\frac{D-1}{2}\right) / \Gamma\!\left(\frac{D}{2}\right)} = \frac{\Gamma\!\left(\frac{D}{2}\right)}{\sqrt{\pi} \;\Gamma\!\left(\frac{D+1}{2}\right)}.
\]
Thus,
\[
\mathbb{E}[|q_1|] = \frac{\Gamma\!\left(\frac{D}{2}\right)}{\sqrt{\pi} \;\Gamma\!\left(\frac{D+1}{2}\right)},
\]
and multiplying by $D$ gives the desired result. $\square$

## 4. Extension to Normalized Vectors

Consider a fixed center $\mathbf{c} \in \mathbb{R}^D$ and a random vector $\mathbf{x}$ such that $\mathbf{x} - \mathbf{c}$ has an isotropic distribution (e.g., a multivariate normal with covariance matrix $\sigma^2 I$). Define the normalized vector
\[
\hat{\mathbf{x}} = \frac{\mathbf{x} - \mathbf{c}}{\|\mathbf{x} - \mathbf{c}\|}.
\]

**Corollary 1.** If $\mathbf{x} - \mathbf{c}$ is isotropic, then $\hat{\mathbf{x}}$ is uniformly distributed on $S^{D-1}$. Consequently,
\[
\mathbb{E}\left[\sum_{i=1}^D |\hat{x}_i|\right] = \frac{D \,\Gamma\!\left(\frac{D}{2}\right)}{\sqrt{\pi} \;\Gamma\!\left(\frac{D+1}{2}\right)}.
\]

*Proof.* The isotropy of $\mathbf{x} - \mathbf{c}$ implies that its direction is uniformly distributed over the unit sphere. Hence, $\hat{\mathbf{x}}$ is uniformly distributed on $S^{D-1}$, and Theorem 1 applies directly. $\square$

## 5. Conclusion

We have derived a closed-form expression for the expected sum of absolute coordinates of a uniformly distributed vector on the $D$-dimensional unit sphere. The result, expressed in terms of the Gamma function, generalizes to any normalized vector whose centered version is isotropic.
