# Face detection helper

`dghs_runner.py` is the model-independent batch adapter used by
`shared/anime_face_collect.py`. On Jetson it runs inside an isolated Python
environment because `dghs-imgutils` conflicts with the SD1.5 training stack.

This helper may be reused by SD1.5, FLUX and future backends. It does not define
or change the dataset's category folders.
