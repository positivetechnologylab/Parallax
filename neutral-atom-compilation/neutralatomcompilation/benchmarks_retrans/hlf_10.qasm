OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
cz q[1],q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cz q[3],q[2];
u3(pi/2,0,pi) q[2];
cz q[2],q[1];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
cz q[1],q[2];
u3(pi/2,0,pi) q[1];
cz q[0],q[1];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cz q[6],q[5];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[8];
cz q[8],q[7];
cz q[7],q[6];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[9];
cz q[9],q[4];
cz q[3],q[4];
u3(pi/2,0,3*pi/2) q[3];
u3(pi/2,0,pi) q[4];
cz q[8],q[9];
cz q[4],q[9];
u3(pi/2,0,pi) q[4];
cz q[8],q[7];
u3(pi/2,0,pi) q[7];
cz q[7],q[6];
u3(pi/2,0,pi) q[7];
cz q[8],q[7];
u3(pi/2,0,pi) q[7];
cz q[7],q[6];
cz q[5],q[6];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cz q[6],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cz q[5],q[6];
u3(pi/2,0,pi) q[5];
cz q[0],q[5];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
cz q[6],q[7];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[9];
cz q[9],q[4];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[9];
cz q[4],q[9];
u3(pi/2,0,pi) q[4];
cz q[3],q[4];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[4];
cz q[4],q[3];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[4];
cz q[3],q[4];
cz q[3],q[2];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cz q[2],q[3];
u3(pi/2,-3*pi/2,pi/2) q[2];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[4];
cz q[8],q[9];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cz q[9],q[8];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cz q[8],q[9];
u3(pi/2,0,pi) q[8];
cz q[7],q[8];
u3(pi/2,0,pi) q[7];
cz q[6],q[7];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
cz q[7],q[8];
u3(pi/2,0,pi) q[7];
cz q[6],q[7];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
cz q[7],q[6];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
cz q[6],q[7];
u3(pi/2,0,pi) q[6];
cz q[1],q[6];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[6];
cz q[6],q[1];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[6];
cz q[1],q[6];
u3(pi/2,2*pi,0) q[1];
cz q[1],q[2];
u3(pi/2,pi/2,0) q[1];
u3(pi/2,-pi/2,0) q[2];
cz q[1],q[2];
u3(pi/2,pi/2,pi/2) q[1];
cz q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(pi/2,pi/2,-pi) q[2];
u3(pi/2,0,pi) q[6];
cz q[6],q[5];
cz q[6],q[1];
u3(pi/2,0,pi) q[1];
u3(0,1.406583,0.16421336) q[6];
cz q[5],q[6];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cz q[6],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cz q[5],q[6];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
cz q[7],q[8];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[8];
cz q[8],q[7];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[8];
cz q[7],q[8];
u3(pi/2,0,pi) q[7];
cz q[6],q[7];
u3(pi/2,0,3*pi/2) q[6];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[9];
cz q[9],q[8];
cz q[3],q[8];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,3*pi/2) q[8];
u3(pi/2,0,3*pi/2) q[9];
