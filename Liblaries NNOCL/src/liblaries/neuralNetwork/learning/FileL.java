package liblaries.neuralNetwork.learning;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import liblaries.neuralNetwork.errors.FileVersionException;
import liblaries.neuralNetwork.functions.FunctionList;

public class FileL {
	public static void saveLS(String fileName,LearningSeqence[] data) throws IOException{
		DataOutputStream out=new DataOutputStream(new FileOutputStream(fileName+".LS"));
		
		out.writeByte(-128);							//File version
		
		out.writeInt(data.length);
		out.writeInt(data[0].inputs.length);
		out.writeInt(data[0].outputs.length);
		
		for(LearningSeqence element:data){
			for(float input:element.inputs){
				out.writeDouble(input);
			}
			for(float wyjœcie:element.outputs){
				out.writeDouble(wyjœcie);
			}
		}
		
		out.close();
	}
	public static LearningSeqence[] readLS(String fileName) throws FileVersionException,IOException{
		DataInputStream in=new DataInputStream(new FileInputStream(fileName+".LS"));		
		
		byte version=in.readByte();
		switch(version){											//Version
		case -128:
			int elementNumber=in.readInt();
			int inputNumber=in.readInt();
			int outputNumber=in.readInt();
			
			LearningSeqence[] Return=new LearningSeqence[elementNumber];
			
			for(int i=0;i<elementNumber;i++){
				Return[i].inputs=new float[inputNumber];
				Return[i].outputs=new float[outputNumber];
				
				for(int j=0;j<inputNumber;j++){
					Return[i].inputs[j]=(float) in.readDouble();
				}
				for(int j=0;j<outputNumber;j++){
					Return[i].outputs[j]=(float) in.readDouble();
				}
			}
			in.close();
			return Return;
		default:in.close();throw new FileVersionException("Don't support file verion newer then -128. This file version: "+version);
		}
	}

	public static LNetwork readLNetwork(String fileName) throws FileVersionException,IOException{
		LNetwork network=new LNetwork();
		
		DataInputStream in=new DataInputStream(new FileInputStream(fileName+".NN"));
		
		byte version=in.readByte();											//file version
		switch(version){
		case -128:
			network.setFunction(FunctionList.getFunction(in.readByte()));
			
			byte layersNumber=in.readByte();
			int inputNumber=in.readInt();
			network.setInputNumber(inputNumber);
			
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
			
			network.setWeights(weights);
			
			break;
			default:in.close();throw new FileVersionException("Don't support file verion newer then -128. This file version: "+version);
		}
		in.close();
			
		return network;
	}
	public static void saveLNetwork(String fileName,LNetwork sieæ) throws IOException{
		File file=new File(fileName+".NN");
		file.createNewFile();
		DataOutputStream save=new DataOutputStream(new FileOutputStream(file));
		
		save.writeByte(-128);											//version of .NN
		
		save.writeByte(sieæ.getFunction().getFunctionID());
		
		float[][][] weights=sieæ.getWeights();
		save.writeByte(weights.length);
		
		save.writeInt(sieæ.getInputNumber());
		
		for(int i=0;i<weights.length;i++){
			save.writeInt(weights[i].length);
		}
		for(int i=0;i<weights.length;i++){
			for(int j=0;j<weights[i].length;j++){
				for(int k=0;k<weights[i][j].length;k++){
					save.writeDouble(weights[i][j][k]);
				}
			}
		}
		save.close();
	}
}