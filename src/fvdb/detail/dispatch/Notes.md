# General ideas about compile-time permutation spaces

Start with the simplest version - some number of axes, each one is characterized by a single
size_t Ni representing extent. So the total permutation space, linearized is the product of
all the Ni's.

Given N0, N1, ... Ni,

You can go from a set of selections on each axis v0, v1, ... vi to a linear index idx
easily. You can go from linear index to v0, v1, ... vi easily just using strides.

how to subspace?

subspace linear index (size_t) space::-> subspace coord (size_t... Is)
subspace coord (size_t... Is) subspace::-> point value pack (auto... Vs)
point value pack (auto... Vs) space::-> space coord (size_t... Is)
space coord (size_t... Is) space::-> space linear index (size_t)


