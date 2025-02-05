"This is a code about a Diffusion model using MNIST Dataset"

INFO 

How to Create Virtual Environment?
Commands
1. python3 -m venv "environmnet name" (create)
2. source "environmnet name"/bin/activate (activate)
3. conda deactivate (deactivate)
4. pip list (installed packages in virtual environment)

Requirments download in V.E
1. pip install -r requirements.txt

Move all the packages that have been installed in V.E to requirements.txt
1. pip freeze > requirements.txt
-> pip freeze is a list of packages that has been installed in V.E

Select certain package installed in V.E and move to requirements.txt
1. pip freeze | grep 'torchvision' > requirements.txt
conda deactivate -> base 환경을 비활성화
deactivate -> 가상환경 비활성화
