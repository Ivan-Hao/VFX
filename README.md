# NTU VFX 2020
```
HDR image
   |-image alignment
   |-HDR reconstrution
   |-tone mapping
```
```
usage: HDR.py [-h] [--alignment ALIGNMENT] [--lambda_ LAMBDA_] [--path PATH]
              [--index INDEX] [--time TIME] [--gamma GAMMA]

optional arguments:
  -h, --help            show this help message and exit
  --alignment ALIGNMENT
                        please do it first and re-run the program with result
                        image dir
  --lambda_ LAMBDA_     function smooth parameter
  --path PATH           image dir path(image only)
  --index INDEX         The based picture's index(>=0)
  --time TIME           The explosure time file path(.txt or .npy)
  --gamma GAMMA         gamma parameter
  ```