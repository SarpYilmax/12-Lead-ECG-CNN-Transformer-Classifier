import matplotlib.pyplot as plt
import numpy as np

from ecg.utils import DEVICE


def generate_saliency_maps(model, loader, cfg: dict, tag: str):
    print("\n  [INFO] Generating saliency maps (one example per class)...")

    found_classes = set()
    target_count = cfg["num_classes"]

    for x_batch, y_batch in loader:
        if len(found_classes) == target_count:
            break

        for i in range(x_batch.size(0)):
            true_class_idx = y_batch[i].item()

            if true_class_idx in found_classes:
                continue

            found_classes.add(true_class_idx)
            class_name = cfg["class_names"][true_class_idx]

            x_single = x_batch[i:i + 1].to(DEVICE)
            x_single.requires_grad_()

            logits_sal = model(x_single)
            pred_class_idx = logits_sal.argmax(dim=1).item()
            pred_name = cfg["class_names"][pred_class_idx]

            score = logits_sal[0, pred_class_idx]

            model.zero_grad()
            score.backward()

            saliency_map = x_single.grad.data.abs().squeeze(0).cpu().numpy()
            x_numpy = x_single.detach().squeeze(0).cpu().numpy()

            max_lead_idx = np.argmax(np.max(saliency_map, axis=1))

            plt.figure(figsize=(14, 6))
            plt.subplot(2, 1, 1)
            plt.plot(x_numpy[max_lead_idx], color="#1f77b4", linewidth=1)
            plt.title(f"Original ECG (Lead {max_lead_idx}) | True: {class_name} - Predicted: {pred_name}")
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            plt.plot(saliency_map[max_lead_idx], color="#d62728", linewidth=1)
            plt.fill_between(
                range(len(saliency_map[max_lead_idx])),
                saliency_map[max_lead_idx],
                color="#d62728",
                alpha=0.3,
            )
            plt.title(f"Saliency Map (Regions the model focused on - Lead {max_lead_idx})")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = f"saliency_map_{tag}_{class_name.replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300)
            plt.show()

            print(f"  [OK] Saliency map created for class '{class_name}' ({save_path}).")

            if len(found_classes) == target_count:
                break

    print("\n  [INFO] Saliency maps saved for all classes.\n")
