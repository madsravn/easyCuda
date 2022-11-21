
# Small and easy CUDA 

This is for you who just wants to mess around with CUDA without having to set it up or know too much about it - handy if you want to learn with trial and error, and if you are better at learning hands-on.

## Build process;

`$ sh compile.sh`

`$ ./filter <file_path> <0-4 filter_id value>`

It'll generate a file called `output_image.jpg` in project folder!
And that's it =]

### filter_id possible values;

|filder_id| filter |
|--|--|
| 0 | blur
| 1 | black_and_white
| 2 | grayscale
| 3 | negative
| 4 | detect_yellow

