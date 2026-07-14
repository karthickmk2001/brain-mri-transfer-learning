"""
Creates a small synthetic MRI-like dataset so you can test that the
notebook pipeline runs end to end BEFORE downloading the real data.

This is for pipeline testing only. Use the real Kaggle dataset
(via prepare_data.py) for your actual lab submission.
"""

import os
import random
import numpy as np
from PIL import Image

random.seed(42)
np.random.seed(42)

SIZE = 224


def make_image(has_tumor):
    img = np.zeros((SIZE, SIZE), dtype=np.float32)
    cx, cy = SIZE // 2, SIZE // 2

    # brain-shaped oval
    for y in range(SIZE):
        for x in range(SIZE):
            d = ((x - cx) / 90) ** 2 + ((y - cy) / 110) ** 2
            if d < 1:
                img[y, x] = 0.3 + 0.4 * (1 - d)

    # tissue texture
    img = np.clip(img + np.random.normal(0, 0.08, (SIZE, SIZE)), 0, 1)

    # bright tumour blob
    if has_tumor:
        tx = random.randint(80, 144)
        ty = random.randint(80, 144)
        tr = random.randint(12, 28)
        for y in range(max(0, ty - tr), min(SIZE, ty + tr)):
            for x in range(max(0, tx - tr), min(SIZE, tx + tr)):
                if (x - tx) ** 2 + (y - ty) ** 2 < tr ** 2:
                    img[y, x] = min(1.0, img[y, x] + random.uniform(0.3, 0.6))

    rgb = np.stack([img * 255, img * 220, img * 200], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def main():
    counts = {
        'train': {'tumor': 120, 'no_tumor': 120},
        'test':  {'tumor': 30,  'no_tumor': 30}
    }

    for split, classes in counts.items():
        for cls, n in classes.items():
            folder = os.path.join('brain_mri', split, cls)
            os.makedirs(folder, exist_ok=True)
            for i in range(n):
                img = make_image(cls == 'tumor')
                img.save(os.path.join(folder, 'img_{:04d}.jpg'.format(i)))
            print("  {:<6} {:<10} {} images".format(split, cls, n))

    print()
    print("Synthetic dataset created. Pipeline can now be tested.")
    print("Remember to replace with real Kaggle data before submitting.")


if __name__ == '__main__':
    main()
