
**Theorem.** Let $\(\mathbf{q} \in \mathbb{R}^D\)$ be uniformly distributed on the unit sphere $\(S^{D-1}\)$. Then

$$\[
\mathbb{E}\left[\sum_{i=1}^D |q_i|\right] = \frac{D \,\Gamma\!\left(\frac{D}{2}\right)}{\sqrt{\pi} \;\Gamma\!\left(\frac{D+1}{2}\right)}.
\]$$

*Proof.* By spherical symmetry, $\(\mathbb{E}\left[\sum_{i=1}^D |q_i|\right] = D \,\mathbb{E}[|q_1|]\)$.  
Let $\(Z_1,\dots,Z_D \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)\)$. Then  

$$\[
\mathbf{q} \stackrel{d}{=} \frac{(Z_1,\dots,Z_D)}{\sqrt{\sum_{j=1}^D Z_j^2}}.
\]$$

Thus  

$$\[
\mathbb{E}[|q_1|] = \mathbb{E}\left[\frac{|Z_1|}{\sqrt{\sum_{j=1}^D Z_j^2}}\right].
\]$$

Define $\(U = \frac{Z_1^2}{\sum_{j=1}^D Z_j^2}\)$. Then $\(U \sim \operatorname{Beta}\!\left(\frac12, \frac{D-1}{2}\right)\)$ and $\(|q_1| = \sqrt{U}\)$.  

The $\(s\)-th moment of \(U\)$ is  

$$\[
\mathbb{E}[U^s] = \frac{B\left(\frac12 + s, \frac{D-1}{2}\right)}{B\left(\frac12, \frac{D-1}{2}\right)},
\]$$

where \(B(\cdot,\cdot)\) is the Beta function. For $\(s = \frac12\)$,

$$\[
\mathbb{E}[U^{1/2}] = \frac{\Gamma(1) \Gamma\!\left(\frac{D}{2}\right)}{\Gamma\!\left(\frac{D+1}{2}\right)} \cdot \frac{\Gamma\!\left(\frac{D}{2}\right)}{\Gamma\!\left(\frac12\right) \Gamma\!\left(\frac{D-1}{2}\right)} = \frac{\Gamma\!\left(\frac{D}{2}\right)}{\sqrt{\pi} \;\Gamma\!\left(\frac{D+1}{2}\right)},
\]$$

using $\(B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}\), \(\Gamma(1)=1\), and \(\Gamma(1/2)=\sqrt{\pi}\).$  
Multiplying by $\(D\)$ yields the theorem. ∎

---

**Corollary.** Let$ \(\mathbf{x}, \mathbf{c} \in \mathbb{R}^D\) $with $\(\mathbf{x}\) such that \(\mathbf{x} - \mathbf{c}\) $is isotropic (e.g., $\(\mathbf{x} \sim \mathcal{N}(\mathbf{c}, \sigma^2 I)\))$. Define $\(\hat{\mathbf{x}} = \frac{\mathbf{x} - \mathbf{c}}{\|\mathbf{x} - \mathbf{c}\|}\). Then \(\hat{\mathbf{x}}\)$ is uniform on $\(S^{D-1}\)$, and

$$\[
\mathbb{E}\left[\sum_{i=1}^D |\hat{x}_i|\right] = \frac{D \,\Gamma\!\left(\frac{D}{2}\right)}{\sqrt{\pi} \;\Gamma\!\left(\frac{D+1}{2}\right)}.
\]$$

*Proof.* Isotropy implies the direction is uniform on the sphere. Apply the theorem. ∎
