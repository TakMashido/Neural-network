package liblaries.neuralNetwork.learning;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import liblaries.neuralNetwork.errors.FileVersionException;
import liblaries.neuralNetwork.functions.FunctionList;

public class FileN {
	public static void zapiszCU(String nazwaPliku,LearningSeqence[] dane) throws IOException{
		DataOutputStream out=new DataOutputStream(new FileOutputStream(nazwaPliku+".SNcu"));
		
		out.writeByte(-128);							//Wersja pliku
		
		out.writeInt(dane.length);
		out.writeInt(dane[0].inputs.length);
		out.writeInt(dane[0].outputs.length);
		
		for(LearningSeqence element:dane){
			for(double wej�cie:element.inputs){
				out.writeDouble(wej�cie);
			}
			for(double wyj�cie:element.outputs){
				out.writeDouble(wyj�cie);
			}
		}
		
		out.close();
	}
	public static LearningSeqence[] wczytajCU(String name) throws FileVersionException,IOException{
		DataInputStream in=new DataInputStream(new FileInputStream(name+".SNcu"));		
		
		byte version=in.readByte();
		switch(version){											//Version
		case -128:
			int Il_Element�w=in.readInt();
			int Il_Wej��=in.readInt();
			int Il_Wyj��=in.readInt();
			
			LearningSeqence[] Return=new LearningSeqence[Il_Element�w];
			
			for(int i=0;i<Il_Element�w;i++){
				Return[i].inputs=new float[Il_Wej��];
				Return[i].outputs=new float[Il_Wyj��];
				
				for(int j=0;j<Il_Wej��;j++){
					Return[i].inputs[j]=(float) in.readDouble();
				}
				for(int j=0;j<Il_Wyj��;j++){
					Return[i].outputs[j]=(float) in.readDouble();
				}
			}
			in.close();
			return Return;
		default:in.close();throw new FileVersionException("Don't support file verion newer then -128. This file version: "+version);
		}
	}

	public static LNetwork wczytajNSie�(String Nazwa) throws FileVersionException,IOException{
		LNetwork sie�=new LNetwork();
		
		DataInputStream in=new DataInputStream(new FileInputStream(Nazwa+".SN"));
		
		byte version=in.readByte();											//file version
		switch(version){
		case -128:
			sie�.setFunction(FunctionList.getFunction(in.readByte()));
			
			byte layersNumber=in.readByte();
			int inputNumber=in.readInt();
			sie�.setInputNumber(inputNumber);
			
			int[] layersSize=new int[layersNumber];	
			
			for(byte i=0;i<layersNumber;i++){
				layersSize[i]=in.readInt();
			}
			
			float[][][] weights=new float[layersNumber][][];
			
			for(int i=0;i<layersNumber;i++) {
				weights[i]=new float[layersSize[i]][i==0?inputNumber:layersSize[i-1]];
				for(int j=0;j<weights[i].length;i++) {
					for(int k=0;k<weights[i][0].length;k++){
						weights[i][j][k]=(float)in.readDouble();
					}
				}
			}
			
			sie�.setWeights(weights);
			
			break;
			default:in.close();throw new FileVersionException("Don't support file verion newer then -128. This file version: "+version);
		}
		in.close();
			
		return sie�;
	}
	public static void zapiszNSie�(String nazwaPliku,LNetwork sie�) throws IOException{				//zapisuje sie� z klasy uczenie
		File file=new File(nazwaPliku+".SN");
		file.createNewFile();
		DataOutputStream save=new DataOutputStream(new FileOutputStream(file));
		
		save.writeByte(-128);											//wersja .SN
		
		save.writeByte(sie�.getFunction().getFunctionID());					//ID funkcji na wyj�ciu
		
		float[][][] weights=sie�.getWeights();
		save.writeByte(weights.length);									//ilo�� warstw
		
		save.writeInt(sie�.getInputNumber());							//ilo�� wej��
		
		for(int i=0;i<weights.length;i++){								//ilo�� neuron�w w ka�dej warstwie
			save.writeInt(weights[i].length);
		}
		for(int i=0;i<weights.length;i++){								//waga ka�dego pol�czenia
			for(int j=0;j<weights[i].length;j++){
				for(int k=0;k<weights[i][j].length;k++){
					save.writeDouble(weights[i][j][k]);
				}
			}
		}
		save.close();
	}
}