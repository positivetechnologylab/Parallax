���      ]�(}�(K K	K��KKK��KKK��KKK��KKK	��KKK��KKK��KKK��KKK��K	K	K��K
KK��KKK��KKK��KKK��KKK��KK	K��KK
K��KKK��KKK��KKK��KKK��KKK��KKK��KKK��KKK��u]�(X�  OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
u3(pi/2,0,-pi) q[18];
u3(pi/2,-pi/2,pi/2) q[13];
u3(pi/2,pi/2,-pi/2) q[11];
u3(pi/2,-pi/2,pi/2) q[12];
u3(pi/2,-pi/2,pi/2) q[6];
u3(pi/2,-pi/2,pi/2) q[9];
u3(pi/2,pi/2,-pi/2) q[3];
u3(pi/2,-pi/2,pi/2) q[1];
u3(pi/2,pi/2,-pi/2) q[20];
u3(pi/2,-pi/2,pi/2) q[24];
u3(pi/2,pi/2,-pi/2) q[21];
u3(pi/2,pi/2,-pi/2) q[17];
u3(pi/2,pi/2,-pi/2) q[23];
u3(0.6905751535897943,0,0) q[8];
u3(1.9789904535897929,-pi,0) q[19];
u3(1.1774729535897936,0,0) q[22];
u3(2.3986476,0,-pi) q[10];
u3(2.6804647999999998,0,-pi) q[16];
u3(0.7933864535897933,0,0) q[7];
u3(0.17065065358979406,0,0) q[14];
u3(1.9075144535897928,-pi,0) q[5];
u3(1.3153054,-pi,-pi) q[15];
u3(0.5563469499999997,-pi,-pi) q[4];
u3(1.602729800000001,0,-pi) q[2];
u3(0.8484668535897933,0,0) q[0];
cz q[13],q[8];
cz q[11],q[10];
cz q[12],q[7];
cz q[6],q[5];
cz q[9],q[4];
cz q[3],q[2];
cz q[1],q[0];
cz q[20],q[19];
cz q[24],q[22];
cz q[21],q[16];
cz q[17],q[14];
u3(pi/4,pi/2,0) q[8];
u3(pi/2,pi/2,-0.6373281732051037) q[13];
u3(pi/2,-pi/2,-2.40067848038469) q[11];
u3(3*pi/4,-pi/2,-pi) q[10];
u3(pi/2,pi/2,-0.3896853732051029) q[12];
u3(pi/4,pi/2,0) q[7];
u3(pi/2,pi/2,0.23170662679489729) q[6];
u3(pi/4,pi/2,-pi) q[5];
u3(3*pi/4,-pi/2,-pi) q[4];
u3(pi/2,pi/2,-1.7425839767948954) q[9];
u3(pi/2,-pi/2,3.135833726794898) q[3];
u3(3*pi/4,-pi/2,-pi) q[2];
u3(pi/2,pi/2,0.2632929267948967) q[1];
u3(pi/4,pi/2,0) q[0];
u3(pi/2,-pi/2,-1.171128676794898) q[20];
u3(pi/4,pi/2,0) q[19];
u3(pi/4,pi/2,0) q[22];
u3(pi/2,pi/2,-0.060160273205101866) q[24];
u3(pi/2,-pi/2,-2.3343366803846877) q[21];
u3(3*pi/4,-pi/2,-pi) q[16];
u3(pi/2,-pi/2,1.4159499732051053) q[17];
u3(pi/4,pi/2,-pi) q[14];
cz q[18],q[8];
u3(pi/4,-pi/2,pi/2) q[8];
cz q[23],q[15];
u3(pi/2,-pi/2,2.7812458267948976) q[23];
u3(3*pi/4,-pi/2,0) q[15];
cz q[13],q[8];
u3(pi/2,0,-3*pi/4) q[13];
u3(pi/4,pi/2,-pi/2) q[8];
cz q[18],q[8];
u3(pi/4,-pi/2,pi/2) q[8];
cz q[18],q[13];
u3(0,0,pi/4) q[18];
u3(pi/4,pi/2,-pi/2) q[13];
cz q[18],q[13];
u3(0,1.4065829705916304,-1.4065829705916302) q[13];
cz q[8],q[10];
cz q[18],q[13];
u3(pi/4,-pi/2,pi/2) q[10];
u3(pi/2,0,pi) q[13];
cz q[11],q[10];
u3(pi/2,0,-3*pi/4) q[11];
u3(pi/4,pi/2,-pi/2) q[10];
cz q[8],q[10];
u3(pi/4,-pi/2,pi/2) q[10];
cz q[9],q[11];
u3(0,0,pi/4) q[9];
u3(pi/4,pi/2,-pi/2) q[11];
cz q[9],q[11];
u3(0,1.4065829705916304,-1.4065829705916302) q[11];
cz q[9],q[7];
u3(pi/4,-pi/2,pi/2) q[7];
cz q[10],q[11];
u3(pi/2,0,pi) q[11];
cz q[12],q[7];
u3(pi/2,0,-3*pi/4) q[12];
u3(pi/4,pi/2,-pi/2) q[7];
cz q[9],q[7];
u3(pi/4,-pi/2,pi/2) q[7];
cz q[14],q[12];
u3(0,0,pi/4) q[14];
u3(pi/4,pi/2,-pi/2) q[12];
cz q[14],q[12];
u3(0,1.4065829705916304,-1.4065829705916302) q[12];
cz q[7],q[5];
u3(pi/4,-pi/2,pi/2) q[5];
cz q[14],q[12];
u3(pi/2,0,pi) q[12];
cz q[6],q[5];
u3(pi/2,0,-3*pi/4) q[6];
u3(pi/4,pi/2,-pi/2) q[5];
cz q[7],q[5];
u3(pi/4,-pi/2,pi/2) q[5];
cz q[7],q[6];
u3(0,0,pi/4) q[7];
u3(pi/4,pi/2,-pi/2) q[6];
cz q[7],q[6];
u3(0,1.4065829705916304,-1.4065829705916302) q[6];
cz q[2],q[4];
cz q[5],q[6];
u3(pi/4,-pi/2,pi/2) q[4];
u3(pi/2,0,pi) q[6];
cz q[9],q[4];
u3(pi/2,0,-3*pi/4) q[9];
u3(pi/4,pi/2,-pi/2) q[4];
cz q[2],q[4];
u3(pi/4,-pi/2,pi/2) q[4];
cz q[2],q[9];
u3(0,0,pi/4) q[2];
u3(pi/4,pi/2,-pi/2) q[9];
cz q[2],q[9];
u3(0,1.4065829705916304,-1.4065829705916302) q[9];
cz q[2],q[7];
u3(pi/4,-pi/2,pi/2) q[7];
cz q[4],q[9];
u3(pi/2,0,pi) q[9];
cz q[3],q[1];
u3(pi/2,0,-3*pi/4) q[3];
u3(pi/4,pi/2,-pi/2) q[1];
cz q[2],q[1];
u3(pi/4,-pi/2,pi/2) q[1];
cz q[2],q[3];
u3(0,0,pi/4) q[2];
u3(pi/4,pi/2,-pi/2) q[3];
cz q[2],q[3];
u3(0,1.4065829705916304,-1.4065829705916302) q[3];
cz q[2],q[0];
u3(pi/4,-pi/2,pi/2) q[0];
cz q[7],q[0];
u3(pi/2,0,-3*pi/4) q[7];
u3(pi/4,pi/2,-pi/2) q[0];
cz q[1],q[3];
u3(pi/2,0,pi) q[3];
cz q[2],q[0];
u3(pi/4,-pi/2,pi/2) q[0];
cz q[2],q[7];
u3(0,0,pi/4) q[2];
u3(pi/4,pi/2,-pi/2) q[7];
cz q[2],q[7];
u3(0,1.4065829705916304,-1.4065829705916302) q[7];
cz q[14],q[19];
cz q[0],q[7];
u3(pi/4,-pi/2,pi/2) q[19];
u3(pi/2,0,pi) q[7];
cz q[20],q[19];
u3(pi/2,0,-3*pi/4) q[20];
u3(pi/4,pi/2,-pi/2) q[19];
cz q[14],q[19];
u3(pi/4,-pi/2,pi/2) q[19];
cz q[18],q[20];
u3(0,0,pi/4) q[18];
u3(pi/4,pi/2,-pi/2) q[20];
cz q[18],q[20];
u3(0,1.4065829705916304,-1.4065829705916302) q[20];
cz q[18],q[22];
u3(pi/4,-pi/2,pi/2) q[22];
cz q[19],q[20];
u3(pi/2,0,pi) q[20];
cz q[24],q[22];
u3(pi/2,0,-3*pi/4) q[24];
u3(pi/4,pi/2,-pi/2) q[22];
cz q[18],q[22];
u3(pi/4,-pi/2,pi/2) q[22];
cz q[18],q[24];
u3(0,0,pi/4) q[18];
u3(pi/4,pi/2,-pi/2) q[24];
cz q[18],q[24];
u3(0,1.4065829705916304,-1.4065829705916302) q[24];
cz q[18],q[16];
u3(pi/4,-pi/2,pi/2) q[16];
cz q[22],q[24];
u3(pi/2,0,pi) q[24];
cz q[21],q[16];
u3(pi/2,0,-3*pi/4) q[21];
u3(pi/4,pi/2,-pi/2) q[16];
cz q[18],q[16];
u3(pi/4,-pi/2,pi/2) q[16];
cz q[18],q[21];
u3(0,0,pi/4) q[18];
u3(pi/4,pi/2,-pi/2) q[21];
cz q[18],q[21];
u3(0,1.4065829705916304,-1.4065829705916302) q[21];
cz q[18],q[8];
u3(pi/4,-pi/2,pi/2) q[8];
cz q[16],q[21];
u3(pi/2,0,pi) q[21];
cz q[13],q[8];
u3(pi/2,0,-3*pi/4) q[13];
u3(pi/4,pi/2,-pi/2) q[8];
cz q[18],q[8];
u3(pi/4,-pi/2,pi/2) q[8];
cz q[18],q[13];
u3(0,0,pi/4) q[18];
u3(pi/4,pi/2,-pi/2) q[13];
cz q[18],q[13];
u3(0,1.4065829705916304,-1.4065829705916302) q[13];
cz q[18],q[15];
u3(pi/4,-pi/2,pi/2) q[15];
cz q[8],q[13];
u3(pi/2,0,pi) q[13];
cz q[23],q[15];
u3(pi/2,0,-3*pi/4) q[23];
u3(pi/4,pi/2,-pi/2) q[15];
cz q[18],q[15];
u3(pi/4,-pi/2,pi/2) q[15];
cz q[18],q[23];
u3(0,0,pi/4) q[18];
u3(pi/4,pi/2,-pi/2) q[23];
cz q[18],q[23];
u3(pi/2,0,-pi) q[18];
u3(0,1.4065829705916304,-1.4065829705916302) q[23];
cz q[15],q[23];
u3(pi/2,0,pi) q[23];�K
ee.