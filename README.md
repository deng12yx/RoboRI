# RoboRI

RoboRI implementation modules are developed on separate branches before review and integration.

| Module | Branch | Scope |
| --- | --- | --- |
| RoboMemArena perturbations and RI-Mem | [`arena/ri-mem-submission-20261124`](https://github.com/deng12yx/RoboRI/tree/arena/ri-mem-submission-20261124) | `robustness_arena_test/submit_codes/` |

`main` currently contains this coordination index. The implementation, executed notebook, validation records, and ten failure / clean-success video pairs are on the module branch above.

## Work from another server

Create a distinct branch from the shared base before adding another module:

```bash
git fetch origin
git switch -c <module>/<server>-<date> origin/main
```

Keep module additions in their own directory where possible. Changes to the Arena submission should use a separate branch and be compared with the Arena branch before merging. Shared-file conflicts require review; separate branches preserve both versions. Do not force-push another server's branch.

## Arena implementation

On the Arena branch, start with [`robustness_arena_test/submit_codes/README.md`](https://github.com/deng12yx/RoboRI/blob/arena/ri-mem-submission-20261124/robustness_arena_test/submit_codes/README.md).

This is the complete prepared submission directory, retaining its original project-relative layout. Native baseline repositories, simulators, training demonstrations, model weights, and existing Python environments are external dependencies; their integration paths and licenses are documented in the submission. The implementation and videos have the validation scope stated there. A full nine-method benchmark and pretrained RI-Mem calibration are not claimed complete.
