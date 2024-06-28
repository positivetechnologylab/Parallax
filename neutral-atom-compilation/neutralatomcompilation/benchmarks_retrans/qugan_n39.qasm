OPENQASM 2.0;
include "qelib1.inc";
qreg q[39];
u3(pi/2,0,-pi) q[0];
u3(1.9263510586291763,1.166173867497573,2.9936033345123905) q[1];
u3(0,-3.082619908120322,1.511823581325423) q[2];
cz q[1],q[2];
u3(pi,-0.8827207976007228,-2.453517124395618) q[1];
u3(1.4276434605408639,0,pi/2) q[2];
cz q[1],q[2];
u3(2.3452202582745647,0,-1.9235692607766484) q[1];
u3(2.3847167303530705,-1.4979734978709969,0.10001370033295176) q[2];
u3(pi,-3.1110298833594077,1.6013590970252825) q[3];
cz q[2],q[3];
u3(pi,-0.8827207976007228,-2.453517124395618) q[2];
u3(1.3818015491681823,0,pi/2) q[3];
cz q[2],q[3];
u3(1.2201223287677685,-pi,pi/2) q[2];
u3(2.5103472798860924,-1.1142542308374201,0.5465660684729312) q[3];
u3(pi,3.0309354324158972,1.4601391056210007) q[4];
cz q[3],q[4];
u3(pi,-0.8827207976007228,-2.453517124395618) q[3];
u3(0.8993473021984502,0,pi/2) q[4];
cz q[3],q[4];
u3(2.0487538990219445,-pi,-pi/2) q[3];
u3(2.027735671898061,1.525725061058985,-0.1018706401285221) q[4];
u3(pi,-3.071842889067919,1.6405460913167698) q[5];
cz q[4],q[5];
u3(pi,-0.8827207976007228,-2.453517124395618) q[4];
u3(1.1744829106546548,0,pi/2) q[5];
cz q[4],q[5];
u3(1.337704696362702,-pi,pi/2) q[4];
u3(0.7564406418611074,3.0136177173951157,-1.4774825912018743) q[5];
u3(0,0.07067336846466965,-1.6414696952595655) q[6];
cz q[5],q[6];
u3(pi,-0.8827207976007228,-2.453517124395618) q[5];
u3(0.9501080163503882,0,pi/2) q[6];
cz q[5],q[6];
u3(0.5559877154235027,0,pi/2) q[5];
u3(0.5071899236766308,2.0416189669346636,2.6142850014393293) q[6];
u3(pi,3.0019471778693294,1.431150851074431) q[7];
cz q[6],q[7];
u3(pi,-0.8827207976007228,-2.453517124395618) q[6];
u3(1.4338245461238637,0,pi/2) q[7];
cz q[6],q[7];
u3(0.49644913589794004,-pi,pi/2) q[6];
u3(1.9607769077507613,1.0646017080255863,2.171905381936792) q[7];
u3(0,-0.9896177343255674,-0.5811785924693296) q[8];
cz q[7],q[8];
u3(pi,-0.8827207976007228,-2.453517124395618) q[7];
u3(1.180145190715271,0,pi/2) q[8];
cz q[7],q[8];
u3(2.8293392069299426,0,-pi/2) q[7];
u3(1.4917714632841328,1.5791717982303695,3.035889952945304) q[8];
u3(0,2.745013011318738,1.9673759690659498) q[9];
cz q[8],q[9];
u3(pi,-0.8827207976007228,-2.453517124395618) q[8];
u3(0.4556665603145209,0,pi/2) q[9];
cz q[8],q[9];
u3(1.4535557352570883,0,pi/2) q[8];
u3(2.110655766184131,-1.2873625024864879,0.5155495452952157) q[9];
u3(0,-1.560984224527141,-0.009812102267754419) q[10];
cz q[9],q[10];
u3(pi,-0.8827207976007228,-2.453517124395618) q[9];
u3(1.1101271656909422,0,pi/2) q[10];
cz q[9],q[10];
u3(0.7611118057456202,0,pi/2) q[9];
u3(1.7841829900708375,-1.277126801528524,-2.1817031877365176) q[10];
u3(0,0.008805027651682451,-1.5796013544465795) q[11];
cz q[10],q[11];
u3(pi,-0.8827207976007228,-2.453517124395618) q[10];
u3(1.0141201046003196,0,pi/2) q[11];
cz q[10],q[11];
u3(2.7649552030863456,0,-pi/2) q[10];
u3(1.6293387265778885,1.5664949648404374,3.068207930587832) q[11];
u3(pi,-3.071842889067919,1.6405460913167698) q[12];
cz q[11],q[12];
u3(pi,-0.8827207976007228,-2.453517124395618) q[11];
u3(1.5430643500034567,0,pi/2) q[12];
cz q[11],q[12];
u3(0.3612427017920011,-pi,pi/2) q[11];
u3(1.284243678328954,2.4124557878192405,-1.3234596585032417) q[12];
u3(0,-0.04478553304148969,-1.5260107937534069) q[13];
cz q[12],q[13];
u3(pi,-0.8827207976007228,-2.453517124395618) q[12];
u3(1.3796873531110132,0,pi/2) q[13];
cz q[12],q[13];
u3(0.07491214924673674,0,pi/2) q[12];
u3(0.3228336789898275,1.608613480720722,3.1017175792218437) q[13];
u3(0,2.9396988737203547,1.772690106664335) q[14];
cz q[13],q[14];
u3(pi,-0.8827207976007228,-2.453517124395618) q[13];
u3(1.5520718941457812,0,pi/2) q[14];
cz q[13],q[14];
u3(0.976680659006153,0,pi/2) q[13];
u3(0.4778528499581641,2.227943226239522,2.4262448379324324) q[14];
u3(pi,2.9595622577090896,1.3887659309141913) q[15];
cz q[14],q[15];
u3(pi,-0.8827207976007228,-2.453517124395618) q[14];
u3(1.3204300775681759,0,pi/2) q[15];
cz q[14],q[15];
u3(0.6643918337568236,-pi,pi/2) q[14];
u3(1.716514585646165,-1.3276557016037418,1.041193827338252) q[15];
u3(pi,-0.1781956451018578,-1.7489919718967535) q[16];
cz q[15],q[16];
u3(pi,-0.8827207976007228,-2.453517124395618) q[15];
u3(0.8240081500062503,0,pi/2) q[16];
cz q[15],q[16];
u3(2.749006693233259,-pi,-pi/2) q[15];
u3(0.8166433642044587,1.7207007609594323,2.9244716146903826) q[16];
u3(0,2.8766925112104236,1.8356964691742652) q[17];
cz q[16],q[17];
u3(pi,-0.8827207976007228,-2.453517124395618) q[16];
u3(1.3151135483417227,0,pi/2) q[17];
cz q[16],q[17];
u3(2.131990647727083,0,-pi/2) q[16];
u3(2.537284400929656,1.386853691874375,2.9192441507115774) q[17];
u3(pi,-3.04493644906626,1.66745253131843) q[18];
cz q[17],q[18];
u3(pi,-0.8827207976007228,-2.453517124395618) q[17];
u3(1.394639358176505,0,pi/2) q[18];
cz q[17],q[18];
u3(0.9463386946560587,-pi,pi/2) q[17];
u3(1.326181995775014,2.671436709074271,-1.448344264889747) q[18];
u3(0,0.013773616913100195,-1.5845699437079972) q[19];
cz q[18],q[19];
u3(pi,-0.8827207976007228,-2.453517124395618) q[18];
u3(0.5182328947661949,0,pi/2) q[19];
cz q[18],q[19];
u3(0.23859549302813385,0,pi/2) q[18];
u3(pi/2,-2.66712723,-pi/2) q[19];
u3(1.9060210738099466,1.5237390950602272,3.126101501143739) q[20];
u3(pi,3.075418741748221,1.5046224149533245) q[21];
cz q[20],q[21];
u3(pi,-0.8827207976007228,-2.453517124395618) q[20];
u3(1.3611561548296809,0,pi/2) q[21];
cz q[20],q[21];
u3(3.07140884051874,-pi/2,1.5288357598827975) q[20];
cz q[1],q[20];
u3(pi/2,pi/2,-pi) q[1];
u3(pi/4,pi/2,0) q[20];
cz q[0],q[20];
u3(pi/4,-pi/2,pi/2) q[20];
cz q[1],q[20];
u3(pi/2,0,-3*pi/4) q[1];
u3(pi/4,pi/2,-pi/2) q[20];
cz q[0],q[20];
cz q[0],q[1];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[1];
cz q[0],q[1];
u3(0,1.4065829705916304,-1.4065829705916302) q[1];
u3(pi/4,-pi/2,pi/2) q[20];
cz q[20],q[1];
u3(pi/2,0,pi) q[1];
u3(2.45628659178305,-0.7101880238434743,-2.1581446891887053) q[21];
u3(pi,-3.071842889067919,1.6405460913167698) q[22];
cz q[21],q[22];
u3(pi,-0.8827207976007228,-2.453517124395618) q[21];
u3(0.9868349985039943,0,pi/2) q[22];
cz q[21],q[22];
u3(2.2977654086597794,pi/2,pi/2) q[21];
cz q[2],q[21];
u3(pi/2,pi/2,-pi) q[2];
u3(pi/4,pi/2,0) q[21];
cz q[0],q[21];
u3(pi/4,-pi/2,pi/2) q[21];
cz q[2],q[21];
u3(pi/2,0,-3*pi/4) q[2];
u3(pi/4,pi/2,-pi/2) q[21];
cz q[0],q[21];
cz q[0],q[2];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[2];
cz q[0],q[2];
u3(0,1.4065829705916304,-1.4065829705916302) q[2];
u3(pi/4,-pi/2,pi/2) q[21];
cz q[21],q[2];
u3(pi/2,0,pi) q[2];
u3(0.8860671385702655,-1.6815684389424153,-2.967508934234262) q[22];
u3(0,2.9396988737203547,1.772690106664335) q[23];
cz q[22],q[23];
u3(pi,-0.8827207976007228,-2.453517124395618) q[22];
u3(1.4616830757443524,0,pi/2) q[23];
cz q[22],q[23];
u3(0.8934179705798838,pi/2,-pi/2) q[22];
cz q[3],q[22];
u3(pi/2,pi/2,-pi) q[3];
u3(pi/4,pi/2,0) q[22];
cz q[0],q[22];
u3(pi/4,-pi/2,pi/2) q[22];
cz q[3],q[22];
u3(pi/2,0,-3*pi/4) q[3];
u3(pi/4,pi/2,-pi/2) q[22];
cz q[0],q[22];
cz q[0],q[3];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[3];
cz q[0],q[3];
u3(0,1.4065829705916304,-1.4065829705916302) q[3];
u3(pi/4,-pi/2,pi/2) q[22];
cz q[22],q[3];
u3(pi/2,0,pi) q[3];
u3(0.4047920979375485,2.594409826246026,2.081284593778433) q[23];
u3(pi,-3.04493644906626,1.66745253131843) q[24];
cz q[23],q[24];
u3(pi,-0.8827207976007228,-2.453517124395618) q[23];
u3(1.37353212160496,0,pi/2) q[24];
cz q[23],q[24];
u3(2.0625446465999158,pi/2,pi/2) q[23];
cz q[4],q[23];
u3(pi/2,pi/2,-pi) q[4];
u3(pi/4,pi/2,0) q[23];
cz q[0],q[23];
u3(pi/4,-pi/2,pi/2) q[23];
cz q[4],q[23];
u3(pi/2,0,-3*pi/4) q[4];
u3(pi/4,pi/2,-pi/2) q[23];
cz q[0],q[23];
cz q[0],q[4];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[4];
cz q[0],q[4];
u3(0,1.4065829705916304,-1.4065829705916302) q[4];
u3(pi/4,-pi/2,pi/2) q[23];
cz q[23],q[4];
u3(pi/2,0,pi) q[4];
u3(2.079140750255745,0.8411175796582713,2.0692035345323863) q[24];
u3(pi,-2.926333756693169,1.7860552236915224) q[25];
cz q[24],q[25];
u3(pi,-0.8827207976007228,-2.453517124395618) q[24];
u3(0.7819335183199972,0,pi/2) q[25];
cz q[24],q[25];
u3(2.1060992211171414,-pi/2,pi/2) q[24];
cz q[5],q[24];
u3(pi/2,pi/2,-pi) q[5];
u3(pi/4,pi/2,0) q[24];
cz q[0],q[24];
u3(pi/4,-pi/2,pi/2) q[24];
cz q[5],q[24];
u3(pi/2,0,-3*pi/4) q[5];
u3(pi/4,pi/2,-pi/2) q[24];
cz q[0],q[24];
cz q[0],q[5];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[5];
cz q[0],q[5];
u3(0,1.4065829705916304,-1.4065829705916302) q[5];
u3(pi/4,-pi/2,pi/2) q[24];
cz q[24],q[5];
u3(pi/2,0,pi) q[5];
u3(1.416957851991058,1.584855662327728,3.050091114216456) q[25];
u3(0,-3.053661019074717,1.482864692279822) q[26];
cz q[25],q[26];
u3(pi,-0.8827207976007228,-2.453517124395618) q[25];
u3(0.5140351801384161,0,pi/2) q[26];
cz q[25],q[26];
u3(0.10374713489162865,-pi/2,-pi/2) q[25];
cz q[6],q[25];
u3(pi/2,pi/2,-pi) q[6];
u3(pi/4,pi/2,0) q[25];
cz q[0],q[25];
u3(pi/4,-pi/2,pi/2) q[25];
cz q[6],q[25];
u3(pi/2,0,-3*pi/4) q[6];
u3(pi/4,pi/2,-pi/2) q[25];
cz q[0],q[25];
cz q[0],q[6];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[6];
cz q[0],q[6];
u3(0,1.4065829705916304,-1.4065829705916302) q[6];
u3(pi/4,-pi/2,pi/2) q[25];
cz q[25],q[6];
u3(pi/2,0,pi) q[6];
u3(1.6421302026552407,1.5560992349686318,-0.20337114263105516) q[26];
u3(0,2.8766925112104236,1.8356964691742652) q[27];
cz q[26],q[27];
u3(pi,-0.8827207976007228,-2.453517124395618) q[26];
u3(1.2150488602346936,0,pi/2) q[27];
cz q[26],q[27];
u3(0.5334252084687839,-pi/2,-pi/2) q[26];
cz q[7],q[26];
u3(pi/2,pi/2,-pi) q[7];
u3(pi/4,pi/2,0) q[26];
cz q[0],q[26];
u3(pi/4,-pi/2,pi/2) q[26];
cz q[7],q[26];
u3(pi/2,0,-3*pi/4) q[7];
u3(pi/4,pi/2,-pi/2) q[26];
cz q[0],q[26];
cz q[0],q[7];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[7];
cz q[0],q[7];
u3(0,1.4065829705916304,-1.4065829705916302) q[7];
u3(pi/4,-pi/2,pi/2) q[26];
cz q[26],q[7];
u3(pi/2,0,pi) q[7];
u3(1.4592368490112495,-2.4651434045750746,-1.6599343912638975) q[27];
u3(pi,-3.04493644906626,1.66745253131843) q[28];
cz q[27],q[28];
u3(pi,-0.8827207976007228,-2.453517124395618) q[27];
u3(1.5173047413966467,0,pi/2) q[28];
cz q[27],q[28];
u3(1.578477007041055,-pi/2,pi/2) q[27];
cz q[8],q[27];
u3(pi/2,pi/2,-pi) q[8];
u3(pi/4,pi/2,0) q[27];
cz q[0],q[27];
u3(pi/4,-pi/2,pi/2) q[27];
cz q[8],q[27];
u3(pi/2,0,-3*pi/4) q[8];
u3(pi/4,pi/2,-pi/2) q[27];
cz q[0],q[27];
cz q[0],q[8];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[8];
cz q[0],q[8];
u3(0,1.4065829705916304,-1.4065829705916302) q[8];
u3(pi/4,-pi/2,pi/2) q[27];
cz q[27],q[8];
u3(pi/2,0,pi) q[8];
u3(1.6019154483961442,0.795900647739868,1.602560250979371) q[28];
u3(pi,3.0309354324158972,1.4601391056210007) q[29];
cz q[28],q[29];
u3(pi,-0.8827207976007228,-2.453517124395618) q[28];
u3(1.4877742120154718,0,pi/2) q[29];
cz q[28],q[29];
u3(1.5744859733127954,-pi/2,pi/2) q[28];
cz q[9],q[28];
u3(pi/2,pi/2,-pi) q[9];
u3(pi/4,pi/2,0) q[28];
cz q[0],q[28];
u3(pi/4,-pi/2,pi/2) q[28];
cz q[9],q[28];
u3(pi/2,0,-3*pi/4) q[9];
u3(pi/4,pi/2,-pi/2) q[28];
cz q[0],q[28];
cz q[0],q[9];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[9];
cz q[0],q[9];
u3(0,1.4065829705916304,-1.4065829705916302) q[9];
u3(pi/4,-pi/2,pi/2) q[28];
cz q[28],q[9];
u3(pi/2,0,pi) q[9];
u3(1.8790147763707963,1.5054491350018946,-0.212461788531642) q[29];
u3(0,-2.934993854656285,1.3641975278613883) q[30];
cz q[29],q[30];
u3(pi,-0.8827207976007228,-2.453517124395618) q[29];
u3(0.7862581435602554,0,pi/2) q[30];
cz q[29],q[30];
u3(0.2825558543925171,pi/2,-pi/2) q[29];
cz q[10],q[29];
u3(pi/2,pi/2,-pi) q[10];
u3(pi/4,pi/2,0) q[29];
cz q[0],q[29];
u3(pi/4,-pi/2,pi/2) q[29];
cz q[10],q[29];
u3(pi/2,0,-3*pi/4) q[10];
u3(pi/4,pi/2,-pi/2) q[29];
cz q[0],q[29];
cz q[0],q[10];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[10];
cz q[0],q[10];
u3(0,1.4065829705916304,-1.4065829705916302) q[10];
u3(pi/4,-pi/2,pi/2) q[29];
cz q[29],q[10];
u3(pi/2,0,pi) q[10];
u3(0.7669041109587849,3.0944370838409307,1.6047633997487054) q[30];
u3(pi,-3.0265869816947917,1.6858019986898984) q[31];
cz q[30],q[31];
u3(pi,-0.8827207976007228,-2.453517124395618) q[30];
u3(1.465368588301331,0,pi/2) q[31];
cz q[30],q[31];
u3(1.6796841501084605,pi/2,pi/2) q[30];
cz q[11],q[30];
u3(pi/2,pi/2,-pi) q[11];
u3(pi/4,pi/2,0) q[30];
cz q[0],q[30];
u3(pi/4,-pi/2,pi/2) q[30];
cz q[11],q[30];
u3(pi/2,0,-3*pi/4) q[11];
u3(pi/4,pi/2,-pi/2) q[30];
cz q[0],q[30];
cz q[0],q[11];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[11];
cz q[0],q[11];
u3(0,1.4065829705916304,-1.4065829705916302) q[11];
u3(pi/4,-pi/2,pi/2) q[30];
cz q[30],q[11];
u3(pi/2,0,pi) q[11];
u3(1.0603401687980827,-1.570872699847857,0.00015631787287073706) q[31];
u3(0,1.9827719657150489,2.729617014669639) q[32];
cz q[31],q[32];
u3(pi,-0.8827207976007228,-2.453517124395618) q[31];
u3(1.5655949414855892,0,pi/2) q[32];
cz q[31],q[32];
u3(0.026216113711196264,-pi/2,-pi/2) q[31];
cz q[12],q[31];
u3(pi/2,pi/2,-pi) q[12];
u3(pi/4,pi/2,0) q[31];
cz q[0],q[31];
u3(pi/4,-pi/2,pi/2) q[31];
cz q[12],q[31];
u3(pi/2,0,-3*pi/4) q[12];
u3(pi/4,pi/2,-pi/2) q[31];
cz q[0],q[31];
cz q[0],q[12];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[12];
cz q[0],q[12];
u3(0,1.4065829705916304,-1.4065829705916302) q[12];
u3(pi/4,-pi/2,pi/2) q[31];
cz q[31],q[12];
u3(pi/2,0,pi) q[12];
u3(1.4789806433147177,-3.018681131783734,-1.5821222932794106) q[32];
u3(0,0.014829897316303242,-1.5856262241111982) q[33];
cz q[32],q[33];
u3(pi,-0.8827207976007228,-2.453517124395618) q[32];
u3(0.5838173370317856,0,pi/2) q[33];
cz q[32],q[33];
u3(1.4935426559716254,-pi/2,-pi/2) q[32];
cz q[13],q[32];
u3(pi/2,pi/2,-pi) q[13];
u3(pi/4,pi/2,0) q[32];
cz q[0],q[32];
u3(pi/4,-pi/2,pi/2) q[32];
cz q[13],q[32];
u3(pi/2,0,-3*pi/4) q[13];
u3(pi/4,pi/2,-pi/2) q[32];
cz q[0],q[32];
cz q[0],q[13];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[13];
cz q[0],q[13];
u3(0,1.4065829705916304,-1.4065829705916302) q[13];
u3(pi/4,-pi/2,pi/2) q[32];
cz q[32],q[13];
u3(pi/2,0,pi) q[13];
u3(2.2810406918627435,0.3163973857391218,1.7811080720620147) q[33];
u3(pi,-0.07491477401882296,-1.6457111008137186) q[34];
cz q[33],q[34];
u3(pi,-0.8827207976007228,-2.453517124395618) q[33];
u3(0.5864974890076129,0,pi/2) q[34];
cz q[33],q[34];
u3(2.2166301062460514,pi/2,pi/2) q[33];
cz q[14],q[33];
u3(pi/2,pi/2,-pi) q[14];
u3(pi/4,pi/2,0) q[33];
cz q[0],q[33];
u3(pi/4,-pi/2,pi/2) q[33];
cz q[14],q[33];
u3(pi/2,0,-3*pi/4) q[14];
u3(pi/4,pi/2,-pi/2) q[33];
cz q[0],q[33];
cz q[0],q[14];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[14];
cz q[0],q[14];
u3(0,1.4065829705916304,-1.4065829705916302) q[14];
u3(pi/4,-pi/2,pi/2) q[33];
cz q[33],q[14];
u3(pi/2,0,pi) q[14];
u3(2.0334699481355516,-1.5359554954408887,-3.063660648037092) q[34];
u3(pi,3.0309354324158972,1.4601391056210007) q[35];
cz q[34],q[35];
u3(pi,-0.8827207976007228,-2.453517124395618) q[34];
u3(1.4250445323829126,0,pi/2) q[35];
cz q[34],q[35];
u3(2.6933783640581224,pi/2,pi/2) q[34];
cz q[15],q[34];
u3(pi/2,pi/2,-pi) q[15];
u3(pi/4,pi/2,0) q[34];
cz q[0],q[34];
u3(pi/4,-pi/2,pi/2) q[34];
cz q[15],q[34];
u3(pi/2,0,-3*pi/4) q[15];
u3(pi/4,pi/2,-pi/2) q[34];
cz q[0],q[34];
cz q[0],q[15];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[15];
cz q[0],q[15];
u3(0,1.4065829705916304,-1.4065829705916302) q[15];
u3(pi/4,-pi/2,pi/2) q[34];
cz q[34],q[15];
u3(pi/2,0,pi) q[15];
u3(1.5906783528438166,-1.5571886566569497,0.6002528323404595) q[35];
u3(pi,-0.010879579775894932,-1.5816759065707897) q[36];
cz q[35],q[36];
u3(pi,-0.8827207976007228,-2.453517124395618) q[35];
u3(1.2929480064984116,0,pi/2) q[36];
cz q[35],q[36];
u3(1.9520122156696942,-pi/2,pi/2) q[35];
cz q[16],q[35];
u3(pi/2,pi/2,-pi) q[16];
u3(pi/4,pi/2,0) q[35];
cz q[0],q[35];
u3(pi/4,-pi/2,pi/2) q[35];
cz q[16],q[35];
u3(pi/2,0,-3*pi/4) q[16];
u3(pi/4,pi/2,-pi/2) q[35];
cz q[0],q[35];
cz q[0],q[16];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[16];
cz q[0],q[16];
u3(0,1.4065829705916304,-1.4065829705916302) q[16];
u3(pi/4,-pi/2,pi/2) q[35];
cz q[35],q[16];
u3(pi/2,0,pi) q[16];
u3(1.39506560151633,1.868478079262518,2.088748734951266) q[36];
u3(pi,3.0309354324158972,1.4601391056210007) q[37];
cz q[36],q[37];
u3(pi,-0.8827207976007228,-2.453517124395618) q[36];
u3(1.483377004435541,0,pi/2) q[37];
cz q[36],q[37];
u3(1.62360215907308,-pi/2,pi/2) q[36];
cz q[17],q[36];
u3(pi/2,pi/2,-pi) q[17];
u3(pi/4,pi/2,0) q[36];
cz q[0],q[36];
u3(pi/4,-pi/2,pi/2) q[36];
cz q[17],q[36];
u3(pi/2,0,-3*pi/4) q[17];
u3(pi/4,pi/2,-pi/2) q[36];
cz q[0],q[36];
cz q[0],q[17];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[17];
cz q[0],q[17];
u3(0,1.4065829705916304,-1.4065829705916302) q[17];
u3(pi/4,-pi/2,pi/2) q[36];
cz q[36],q[17];
u3(pi/2,0,pi) q[17];
u3(1.4682061796378831,1.5809151502004593,-0.0984903776906969) q[37];
u3(pi,-0.05354836323463852,-1.6243446900295346) q[38];
cz q[37],q[38];
u3(pi,-0.8827207976007228,-2.453517124395618) q[37];
u3(1.4149807351620576,0,pi/2) q[38];
cz q[37],q[38];
u3(2.5770179628436134,pi/2,pi/2) q[37];
cz q[18],q[37];
u3(pi/2,pi/2,-pi) q[18];
u3(pi/4,pi/2,0) q[37];
cz q[0],q[37];
u3(pi/4,-pi/2,pi/2) q[37];
cz q[18],q[37];
u3(pi/2,0,-3*pi/4) q[18];
u3(pi/4,pi/2,-pi/2) q[37];
cz q[0],q[37];
cz q[0],q[18];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[18];
cz q[0],q[18];
u3(0,1.4065829705916304,-1.4065829705916302) q[18];
u3(pi/4,-pi/2,pi/2) q[37];
cz q[37],q[18];
u3(pi/2,0,pi) q[18];
u3(0.4150636000000005,-pi,-pi) q[38];
cz q[19],q[38];
u3(pi/2,pi/2,-pi) q[19];
u3(pi/4,pi/2,0) q[38];
cz q[0],q[38];
u3(pi/4,-pi/2,pi/2) q[38];
cz q[19],q[38];
u3(pi/2,0,-3*pi/4) q[19];
u3(pi/4,pi/2,-pi/2) q[38];
cz q[0],q[38];
cz q[0],q[19];
u3(0,0,pi/4) q[0];
u3(pi/4,pi/2,-pi/2) q[19];
cz q[0],q[19];
u3(pi/2,0,-pi) q[0];
u3(0,1.4065829705916304,-1.4065829705916302) q[19];
u3(pi/4,-pi/2,pi/2) q[38];
cz q[38],q[19];
u3(pi/2,0,pi) q[19];
