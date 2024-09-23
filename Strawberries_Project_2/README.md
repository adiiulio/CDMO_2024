
to run the docker file for SMT:
0) need to be in the SMT folder (navigate there if necessary)
1) docker build . -t image -f DockerFile
2) docker run -it -v "$(pwd)/data:/src/data" image

to run the docker file for CP:
1) cd CP
2) docker build . -t image -f Dockerfile
3) docker run -it -v "$(pwd)/instances:/src/instances" image

to run the docker file for MIP:
1) cd MIP
2) docker build . -t image -f Dockerfile
3) docker run -ti image

to run the solution checker: 
cd checker
python (or python3) check_solution.py instances/ results/