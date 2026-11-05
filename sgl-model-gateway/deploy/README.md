# Deployment manifests

Manifests for running `sgl-model-gateway` with the tenancy feature on
Kubernetes. Apply order:

    kubectl -n infer apply -f tenancy-rbac.yaml      # once per namespace
    kubectl -n infer apply -f tenancy-router.yaml    # the workload

`tenancy-rbac.yaml` owns the ServiceAccount and its two Roles, so it is the only
file that creates them; `tenancy-router.yaml` references the account by name.
Skip the RBAC file when running under an account that already exists, and point
`serviceAccountName` at it after granting it both Roles.

Re-applying `tenancy-router.yaml` does not disturb live state: the alloc
ConfigMap declares no `data`, so allocations the gateway wrote to it survive.

| File | Contents |
|---|---|
| `tenancy-rbac.yaml` | ServiceAccount, ConfigMap Role, pod-discovery Role, bindings |
| `tenancy-router.yaml` | Key and alloc ConfigMaps, Service, Deployment |
| `tenancy-configmaps.example.yaml` | Standalone ConfigMap examples |

The feature is behind a default-off `tenancy` cargo feature, so a stock image
does not carry it. While it ships as a wheel rather than inside an image, the
Deployment installs that wheel from a host path at startup and is pinned to the
node holding it; both are marked in the file and go away once the feature is
built into an image.

The namespace, worker selector, image and node in `tenancy-router.yaml` are the
values of one deployment — change them for yours.
