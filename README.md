# Expected Value of the $L_1$-norm of a Uniformly Distributed Vector on the Unit Sphere

---

## 1. Problem Formulation

Let $\hat{\mathbf{x}} = [\hat{x}_1, \hat{x}_2, \dots, \hat{x}_D]^T \in \mathbb{R}^D$ be a random vector uniformly distributed on the unit sphere $S^{D-1} = \{ \mathbf{x} \in \mathbb{R}^D : \|\mathbf{x}\|_2 = 1 \}$. We seek to derive the expectation of the $L_1$-norm:

$$E = \mathbb{E} \left[ \|\hat{\mathbf{x}}\|_1 \right] = \mathbb{E} \left[ \sum_{i=1}^D |\hat{x}_i| \right]$$

By the symmetry of the sphere and the linearity of expectation:

$$E = D \cdot \mathbb{E}[|\hat{x}_1|]$$

---

## 2. Core Lemmas

### Lemma 1: Gaussian Representation
A vector $\hat{\mathbf{x}}$ uniformly distributed on $S^{D-1}$ can be generated from $D$ i.i.d. standard normal variables $Z_i \sim \mathcal{N}(0, 1)$ as:
$$\hat{x}_i = \frac{Z_i}{R}, \quad \text{where } R = \sqrt{\sum_{j=1}^D Z_j^2}$$
Crucially, the direction vector $\hat{\mathbf{x}}$ and the magnitude $R$ (radius) are statistically independent.

### Lemma 2: Ratio Expectation via Independence
Given the independence of $\hat{\mathbf{x}}$ and $R$, for any component $i$:
$$\mathbb{E}[|Z_i|] = \mathbb{E}[|\hat{x}_i| \cdot R] = \mathbb{E}[|\hat{x}_i|] \cdot \mathbb{E}[R]$$
Therefore, the marginal expectation is:
$$\mathbb{E}[|\hat{x}_1|] = \frac{\mathbb{E}[|Z_1|]}{\mathbb{E}[R]}$$

---

## 3. Theorem and Proof

**Theorem:** The expected $L_1$-norm of a uniformly distributed vector on the unit sphere $S^{D-1}$ is:
$$E = \frac{D \cdot \Gamma(D/2)}{\sqrt{\pi} \cdot \Gamma((D+1)/2)}$$

**Proof:**
1. **Expectation of $|Z_1|$**:  
   Since $Z_1 \sim \mathcal{N}(0, 1)$, its absolute value follows a folded normal distribution:
   $$\mathbb{E}[|Z_1|] = \int_{-\infty}^{\infty} |z| \frac{1}{\sqrt{2\pi}} e^{-z^2/2} dz = 2 \int_{0}^{\infty} \frac{z}{\sqrt{2\pi}} e^{-z^2/2} dz = \sqrt{\frac{2}{\pi}}$$

2. **Expectation of $R$**:  
   The variable $R$ follows a Chi-distribution $\chi(D)$. Its mean is:
   $$\mathbb{E}[R] = \sqrt{2} \frac{\Gamma((D+1)/2)}{\Gamma(D/2)}$$

3. **Substitution**:  
   Substitute the results from steps 1 and 2 into the relation from Lemma 2:
   $$\mathbb{E}[|\hat{x}_1|] = \frac{\sqrt{2/\pi}}{\sqrt{2} \frac{\Gamma((D+1)/2)}{\Gamma(D/2)}} = \frac{\Gamma(D/2)}{\sqrt{\pi} \Gamma((D+1)/2)}$$

4. **Total Expectation**:  
   Multiplying by the dimension $D$:
   $$E = D \cdot \frac{\Gamma(D/2)}{\sqrt{\pi} \Gamma((D+1)/2)}$$
   $\square$

---

## 4. Asymptotic Analysis

For high-dimensional spaces ($D \to \infty$), we apply Stirling's approximation for the ratio of Gamma functions:
$$\frac{\Gamma(z+a)}{\Gamma(z+b)} \approx z^{a-b}$$
Setting $z = D/2$, $a = 0$, and $b = 1/2$:
$$\frac{\Gamma(D/2)}{\Gamma(D/2 + 1/2)} \approx \left( \frac{D}{2} \right)^{-1/2} = \sqrt{\frac{2}{D}}$$

Substituting this into the Theorem:
$$E \approx \frac{D}{\sqrt{\pi}} \sqrt{\frac{2}{D}} = \sqrt{\frac{2D}{\pi}}$$

> **Observation:** While the maximum possible $L_1$-norm on a unit sphere is $\sqrt{D}$ (achieved at $(\pm 1/\sqrt{D}, \dots, \pm 1/\sqrt{D})$), the average $L_1$-norm is approximately $0.798 \sqrt{D}$.

---