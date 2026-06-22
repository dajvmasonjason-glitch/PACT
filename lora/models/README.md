# Models

This structure follows the KnOTS repo. Here are instructions for adding models.

#### Create architecture file
First, create a file with your model's architecture. You are free to name it anything.

#### Make config parser recognize your model.
Add an `elif` to the `prepare_models` function in `utils.py` that checks for your model name, and a `prepare_<Your-MODEL>` function to handle model preparation (loading pretrained base models and initializing a merged model). Follow the template provided by existing `prepare_<model-name>` functions.
