TO RUN EVERYTHING ENSURE THAT YOU HAVE PYTHON3 INSTALLED

to run the docker file for SMT:
0) need to be in the SMT folder (navigate there if necessary)
1) docker build . -t image -f DockerFile
2) docker run -it -v "$(pwd)/data:/src/data" image

to run solution checker for SMT files run through docker:
0) need to be in the SMT folder
1) 

to run the docker file for CP:
0) need to be in the CP folder (navigate there if necessary)
1) docker build . -t image -f DockerFile
2) docker run -it -v "$(pwd)/instances:/src/instances" image

to run solution checker for CP files run through docker:
0) need to be in the CP folder
1) 

to run the docker file for MIP:

to run the solution checker for the instances run in our local machines: 
1) 

to run the SMT python script:
1) Navigate to the SMT folder
2) run the command python run_instances.py
