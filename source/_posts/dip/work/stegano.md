---
title: steganography
date: 2023-09-15 19:49:33
tags: DIP
---

# 1. stegano package

```sh
pip install stegano
```

demo

```
from stegano import lsb
msg = 'i luv u'
secret_img = lsb.hide(img_path, msg)
secret_img.save(secret_img_path)

decryped_msg = lsb.reveal(secret_img_path)
```

# 2. 