# RoboMME qualitative pairs

The videos pair a successful clean rollout (left) with a failed or timed-out perturbed rollout (right). Instructions and original episode records are in `cases.json`. These are historical qualitative cases; perturbation timing, budgets and displacement differ from the paper protocol.

| Method | Task / episode | Event | Video | Frames |
|---|---|---|---|---|
| perceptual-framesamp-modul | ButtonUnmaskSwap / 2 | `mid_arrive_early` | [MP4](assets/01_perceptual-framesamp-modul_ButtonUnmaskSwap_ep2.mp4) | [JPG](assets/01_perceptual-framesamp-modul_ButtonUnmaskSwap_ep2.jpg) |
| perceptual-tokendrop-modul | ButtonUnmaskSwap / 3 | `on_arrive_sync` | [MP4](assets/02_perceptual-tokendrop-modul_ButtonUnmaskSwap_ep3.mp4) | [JPG](assets/02_perceptual-tokendrop-modul_ButtonUnmaskSwap_ep3.jpg) |
| perceptual-framesamp-modul | VideoUnmask / 0 | `mid_takeback_early` | [MP4](assets/03_perceptual-framesamp-modul_VideoUnmask_ep0.mp4) | [JPG](assets/03_perceptual-framesamp-modul_VideoUnmask_ep0.jpg) |
| simplememvla-robomme | BinFill / 10 | `mid_displace` | [MP4](assets/04_simplememvla-robomme_BinFill_ep10.mp4) | [JPG](assets/04_simplememvla-robomme_BinFill_ep10.jpg) |

The notebook outputs are cleared to avoid embedding duplicate copies of the videos in Git history. Run the notebook to display the local media. The Markdown links above also work directly on GitHub.
