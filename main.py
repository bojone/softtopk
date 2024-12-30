import numpy as np
import jax.numpy as jnp
import jax.lax as lax


def softtopk(x, k):
    """differentiable top-k operator for jax
    Refer: https://kexue.fm/archives/10373#%E4%BA%8C%E8%80%85%E5%85%BC%E4%B9%8B
    """
    x_sort = x.astype('float32').sort(axis=-1)
    lse1 = lax.cumlogsumexp(x_sort, axis=x.ndim - 1)
    lse2 = lax.cumlogsumexp(-x_sort, axis=x.ndim - 1, reverse=True)
    lse2 = jnp.roll(lse2, -1, axis=-1).at[..., -1].set(-jnp.inf)
    km = k - jnp.arange(x.shape[-1] - 1, -1, -1)
    x_lamb = lse1 - jnp.log(jnp.sqrt(km**2 + jnp.exp(lse1 + lse2)) + km)
    x_sort_ = jnp.roll(x_sort, -1, axis=-1).at[..., -1].set(jnp.inf)
    idxs = ((x_lamb <= x_sort_) & (x_lamb >= x_sort)).argmax(axis=-1)
    lamb = jnp.take_along_axis(x_lamb, idxs[..., None], axis=-1)
    p = (1 - np.exp(-np.abs(x - lamb))) * np.sign(x - lamb) * 0.5 + 0.5
    return p.astype(x.dtype)


x = jnp.array(np.random.randn(32, 128))
p = softtopk(x, 16)
print(p.sum(axis=-1))
