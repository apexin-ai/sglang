# Deployment manifests

Manifests for running `sgl-model-gateway` with the tenancy feature on
Kubernetes.

Apply once per namespace, shared by every model:

    kubectl -n infer apply -f tenancy-rbac.yaml    # ServiceAccount + Roles
    kubectl -n infer apply -f tenancy-keys.yaml    # tenant keys

Then one file per model:

    kubectl -n infer apply -f tenancy-router.yaml  # DeepSeek-V4.1 router
    kubectl -n infer apply -f qwen38-tenancy.yaml  # Qwen3.8 workers + router

| File | Contents | Scope |
|---|---|---|
| `tenancy-rbac.yaml` | ServiceAccount, ConfigMap Role, pod-discovery Role, bindings | per namespace |
| `tenancy-keys.yaml` | `smg-tenant-keys`: tenants, keys, roles | per namespace, every model |
| `tenancy-router.yaml` | DeepSeek-V4.1 quota ConfigMap, Service, Deployment | per model |
| `qwen38-tenancy.yaml` | Qwen3.8 workers, quota ConfigMap, router | per model |

## Keys are global, quota is per model

A tenant's key works on every router pointing `--tenancy-keys-configmap` at
`smg-tenant-keys`. Quota lives in each model's own ConfigMap (the one passed as
`--tenancy-alloc-configmap`), under `data["quota.yaml"]`:

    version: 1
    tenants:
      chengsong: {initial: 16, floor: 4, ceiling: 48}

A tenant not listed there is unlimited on that model. The same ConfigMap also
holds `alloc.yaml`, the current allocations the gateway writes back; the
manifests declare only `quota.yaml`, so re-applying never resets allocations,
and the gateway patches only `alloc.yaml`, so it never overwrites the rules.

Quota is observe-only today: an over-cap request is served and counted.

## Notes

The feature is behind a default-off `tenancy` cargo feature, so a stock image
does not carry it. While it ships as a wheel rather than inside an image, each
router installs that wheel at startup from a host path or CPFS; both are marked
in the files and go away once the feature is built into an image.

Services expose only the serving port. The namespace PodMonitor scrapes the
declared container ports directly, so the metrics port stays on the container
and off the Service.
