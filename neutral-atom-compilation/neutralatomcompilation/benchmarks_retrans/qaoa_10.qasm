OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
u3(0,pi/2,-pi/2) q[4];
cz q[0],q[4];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[4];
cz q[4],q[0];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[4];
cz q[0],q[4];
u3(pi/2,-1.5281106482289797,-pi) q[4];
u3(0,pi/2,-pi/2) q[5];
cz q[1],q[5];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[5];
cz q[5],q[1];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[5];
cz q[1],q[5];
u3(pi/2,0,pi) q[1];
cz q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
cz q[1],q[0];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
cz q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
u3(0,pi/2,-pi/2) q[5];
u3(0,pi/2,-pi/2) q[6];
cz q[2],q[6];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[6];
cz q[6],q[2];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[6];
cz q[2],q[6];
u3(pi/2,0,pi) q[6];
u3(0,pi/2,-pi/2) q[7];
cz q[3],q[7];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[7];
cz q[7],q[3];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[7];
cz q[3],q[7];
u3(pi/2,0,pi) q[3];
cz q[6],q[3];
u3(0.6872404599999998,-pi/2,pi/2) q[3];
cz q[6],q[3];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[6];
cz q[2],q[6];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[6];
cz q[6],q[2];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[6];
cz q[2],q[6];
cz q[2],q[5];
u3(0.6872404599999998,-pi/2,pi/2) q[5];
cz q[2],q[5];
u3(pi/2,0,pi) q[5];
u3(0,pi/2,-pi/2) q[6];
cz q[3],q[6];
u3(0.6872404599999998,-pi/2,pi/2) q[6];
cz q[3],q[6];
u3(pi/2,0,pi) q[3];
cz q[2],q[3];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cz q[3],q[2];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cz q[2],q[3];
u3(pi/2,0,pi) q[3];
u3(0,pi/2,-pi/2) q[6];
u3(0,pi/2,-pi/2) q[7];
cz q[3],q[7];
u3(0.6872404599999998,-pi/2,pi/2) q[7];
cz q[3],q[7];
cz q[3],q[6];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[6];
cz q[6],q[3];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[6];
cz q[3],q[6];
u3(pi/2,0,pi) q[3];
u3(0,pi/2,-pi/2) q[6];
u3(0,pi/2,-pi/2) q[7];
u3(0,pi/2,-pi/2) q[8];
cz q[5],q[8];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[8],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[5],q[8];
u3(pi/2,0,pi) q[5];
cz q[2],q[5];
u3(0.6872404599999998,-pi/2,pi/2) q[5];
cz q[2],q[5];
cz q[2],q[3];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cz q[3],q[2];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cz q[2],q[3];
u3(pi/2,0,pi) q[2];
u3(0,pi/2,-pi/2) q[3];
u3(pi/2,0,pi) q[5];
u3(0,pi/2,-pi/2) q[8];
cz q[5],q[8];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[8],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[5],q[8];
cz q[5],q[6];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cz q[6],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cz q[5],q[6];
u3(pi/2,0,pi) q[6];
cz q[6],q[7];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
cz q[7],q[6];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
cz q[6],q[7];
u3(pi/2,0,pi) q[6];
u3(0,pi/2,-pi/2) q[7];
u3(0,pi/2,-pi/2) q[8];
u3(pi/2,0,pi) q[9];
cz q[9],q[8];
u3(0.6872404599999998,-pi/2,pi/2) q[8];
cz q[9],q[8];
u3(0,pi/2,-pi/2) q[8];
cz q[5],q[8];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[8],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[5],q[8];
cz q[5],q[6];
u3(pi/2,0,pi) q[6];
cz q[6],q[7];
u3(pi/2,0,pi) q[6];
cz q[5],q[6];
u3(pi/2,0,pi) q[6];
u3(0,pi/2,-pi/2) q[7];
cz q[6],q[7];
u3(pi/2,0,pi) q[6];
cz q[5],q[6];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cz q[6],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cz q[5],q[6];
u3(pi/2,0,pi) q[5];
cz q[1],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
u3(0.6872404599999998,-pi/2,pi/2) q[7];
cz q[6],q[7];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
cz q[7],q[6];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
cz q[6],q[7];
u3(0.5302568027964133,-pi,pi/2) q[7];
u3(pi,1.2490457723982544,3.0828815691628915) q[8];
cz q[4],q[8];
u3(0,-2.0597393236654105,0.4462573183045979) q[4];
u3(2.4543521935897936,2.6177653267948973,-2.8785531836200517) q[8];
cz q[5],q[8];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[8],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[5],q[8];
cz q[5],q[2];
u3(0,pi/2,-pi/2) q[2];
u3(pi/2,0,pi) q[5];
cz q[4],q[5];
u3(0,0,0.68724046) q[4];
u3(pi/2,0,pi) q[5];
cz q[5],q[2];
u3(0,pi/2,-pi/2) q[2];
u3(pi/2,0,pi) q[5];
cz q[4],q[5];
u3(pi/2,0,pi) q[5];
cz q[5],q[2];
u3(0,pi/2,-pi/2) q[2];
u3(pi/2,0,pi) q[5];
cz q[4],q[5];
u3(pi/2,0,pi) q[5];
cz q[5],q[2];
u3(pi/2,0,pi) q[2];
cz q[2],q[3];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cz q[3],q[2];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cz q[2],q[3];
u3(pi/2,0,pi) q[2];
cz q[1],q[2];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
cz q[2],q[1];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
cz q[1],q[2];
cz q[1],q[0];
u3(0.6872404599999998,-pi/2,pi/2) q[0];
cz q[1],q[0];
u3(pi/2,0,pi) q[0];
u3(1.046969,3*pi/2,pi/2) q[1];
u3(pi/2,0,pi) q[2];
u3(0,pi/2,-pi/2) q[3];
u3(pi/2,0,pi) q[5];
cz q[4],q[5];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cz q[5],q[4];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cz q[4],q[5];
u3(pi/2,0,pi) q[4];
cz q[1],q[4];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[4];
cz q[4],q[1];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[4];
cz q[1],q[4];
u3(0,pi/2,-pi/2) q[4];
cz q[0],q[4];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[4];
cz q[4],q[0];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[4];
cz q[0],q[4];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cz q[6],q[3];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[6];
cz q[3],q[6];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[6];
cz q[6],q[3];
u3(pi/2,-pi/2,1.2606222569645622) q[3];
cz q[3],q[7];
u3(pi/2,0,-pi/2) q[3];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi/2) q[7];
cz q[3],q[7];
u3(pi/2,-0.5302568027964121,0) q[3];
u3(1.6561637861265928,-2.830246897320426,-3.114160620404495) q[7];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cz q[8],q[9];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cz q[9],q[8];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cz q[8],q[9];
u3(pi/2,0,pi) q[9];
cz q[9],q[6];
u3(0,pi/2,-pi/2) q[6];
u3(pi/2,0,pi) q[9];
cz q[5],q[9];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[9];
cz q[9],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[9];
cz q[5],q[9];
u3(pi/2,0,pi) q[5];
cz q[2],q[5];
u3(0,0,0.68724046) q[2];
u3(pi/2,0,pi) q[5];
cz q[5],q[6];
u3(pi/2,0,pi) q[5];
cz q[2],q[5];
u3(pi/2,-pi/2,2.2327977216866266) q[2];
u3(pi/2,0,pi) q[5];
u3(0,pi/2,-pi/2) q[6];
cz q[5],q[6];
u3(2.648356345708767,-pi,-pi/2) q[5];
cz q[2],q[5];
u3(pi/2,0,-pi/2) q[2];
u3(pi/2,0,pi/2) q[5];
cz q[2],q[5];
u3(2.648356345708767,-pi/2,pi/2) q[2];
u3(1.9662296138362607,-0.72896738407298,2.810260690246632) q[5];
u3(0.6872404599999995,-pi/2,pi/2) q[6];
cz q[2],q[6];
u3(pi/2,0,-pi) q[2];
cz q[1],q[2];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
cz q[2],q[1];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
cz q[1],q[2];
u3(pi/2,pi/2,-pi) q[1];
u3(pi/2,0,pi) q[2];
u3(0,1.0103837318112694,0.03658526818873087) q[6];
u3(0,pi/2,-pi/2) q[9];
cz q[8],q[9];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cz q[9],q[8];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cz q[8],q[9];
u3(pi/2,0,pi) q[8];
u3(2.8156033110189416,-pi,-pi/2) q[9];
cz q[5],q[9];
u3(pi/2,0,-pi/2) q[5];
u3(pi/2,0,pi/2) q[9];
cz q[5],q[9];
u3(pi/2,2.8156033110189416,-pi) q[5];
cz q[5],q[8];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[8],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[5],q[8];
cz q[5],q[6];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[8];
u3(1.2557875866221933,0.40829674324175746,-2.5200902647772736) q[9];
cz q[8],q[9];
u3(pi/2,0,pi) q[8];
cz q[5],q[8];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[8],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
cz q[5],q[8];
u3(pi/2,pi/2,-0.2813096027132769) q[5];
cz q[1],q[5];
u3(pi/2,-pi/2,pi/2) q[1];
u3(pi/2,-pi,0) q[5];
cz q[1],q[5];
u3(0.8835558667948968,0,pi/2) q[1];
u3(pi/2,-pi/2,-pi) q[5];
cz q[1],q[5];
u3(pi/2,-1.2894867240816197,pi/2) q[1];
u3(pi/2,-pi,pi/2) q[5];
u3(0,pi/2,-pi/2) q[8];
u3(0,pi/2,-pi/2) q[9];
cz q[6],q[9];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[9];
cz q[9],q[6];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[9];
cz q[6],q[9];
u3(pi/2,0,pi) q[6];
cz q[3],q[6];
u3(0.6872404599999998,-pi/2,pi/2) q[6];
cz q[3],q[6];
u3(1.0469690000000003,-pi/2,pi/2) q[3];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[9];
cz q[9],q[8];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cz q[8],q[9];
u3(0,pi/2,-pi/2) q[9];
cz q[6],q[9];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[9];
cz q[9],q[6];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[9];
cz q[6],q[9];
cz q[6],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cz q[5],q[6];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[6];
cz q[6],q[5];
u3(0,pi/2,-pi/2) q[5];
cz q[1],q[5];
u3(0.6872404599999998,-pi/2,pi/2) q[5];
cz q[1],q[5];
u3(1.046969,3*pi/2,pi/2) q[1];
u3(0,pi/2,-pi/2) q[5];
cz q[4],q[5];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cz q[5],q[4];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cz q[4],q[5];
u3(0,1.9894811131974972,-0.41868478640260065) q[5];
u3(pi/2,pi/2,-pi) q[6];
cz q[6],q[7];
u3(pi/2,-pi/2,pi/2) q[6];
u3(pi/2,-pi,0) q[7];
cz q[6],q[7];
u3(0.8835558667948965,-pi,-pi/2) q[6];
u3(pi/2,-pi/2,-pi) q[7];
cz q[6],q[7];
u3(1.7997700949288016,-pi,0) q[6];
cz q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[5];
u3(pi/2,-pi,0) q[6];
cz q[5],q[6];
u3(0.8835558667948968,0,pi/2) q[5];
u3(pi/2,-pi/2,-pi) q[6];
cz q[5],q[6];
u3(1.431479072999739,2.6177653267948973,-pi) q[5];
cz q[4],q[5];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cz q[5],q[4];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cz q[4],q[5];
u3(0,pi/2,-pi/2) q[5];
u3(pi,-2.1563615517864,-0.5855652249915035) q[6];
cz q[6],q[5];
u3(0.6872404599999998,-pi/2,pi/2) q[5];
cz q[6],q[5];
u3(pi/2,0,-2.0946236535897933) q[5];
u3(2.094623653589794,pi/2,0) q[7];
u3(0,pi/2,-pi/2) q[9];
cz q[6],q[9];
u3(0.6872404599999998,-pi/2,pi/2) q[9];
cz q[6],q[9];
u3(1.046969,3*pi/2,pi/2) q[6];
u3(pi/2,0,-2.0946236535897933) q[9];
