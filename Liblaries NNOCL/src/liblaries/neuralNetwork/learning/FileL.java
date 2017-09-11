package liblaries.neuralNetwork.learning;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import liblaries.neuralNetwork.errors.FileVersionException;
import liblaries.neuralNetwork.errors.NeuralException;
import liblaries.neuralNetwork.functions.Function;
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
		
		Function function;
		int layersNumber;
		int inputNumber;
		int[] layersSize;
		float[][][] weights;
		
		byte version=in.readByte();											//file version
		switch(version){
		case -128:
			function=FunctionList.getFunction(in.readByte());
			
			layersNumber=in.readByte();
			inputNumber=in.readInt();
			
			layersSize=new int[layersNumber];
			
			weights=new float[layersNumber][][];
			for(byte i=0;i<layersNumber;i++){								//Size of each layer
				layersSize[i]=in.readInt();
				weights[i]=new float[layersSize[i]][];
			}
			
			for(int j=0;j<layersSize[0];j++){
				weights[0][j]=new float[inputNumber];
				weights[0][j][0]=0f;
				for(int k=0;k<inputNumber;k++){
					weights[0][j][k+1]=(float) in.readDouble();
				}
			}
			
			if(layersNumber>1){
				for(byte i=1;i<layersNumber;i++){
					weights[i]=new float[layersSize[i]][];
					
					for(int j=0;j<layersSize[i];j++){
						weights[i][j]=new float[layersSize[i-1]+1];
						
						weights[i][j][0]=0f;
						for(int k=0;k<layersSize[i-1];k++){
							weights[i][j][k+1]=(float) in.readDouble();
						}
					}
				}
			}
			network.setWeights(weights);
			network.setFunction(function);
			break;
		case -127:
			function=FunctionList.getFunction(in.readByte());
			
			layersNumber=in.readInt();
			inputNumber=in.readInt();
			
			layersSize=new int[layersNumber];
			
			weights=new float[layersNumber][][];
			for(byte i=0;i<layersNumber;i++){								//Size of each layer
				layersSize[i]=in.readInt();
				weights[i]=new float[layersSize[i]][];
			}
			
			for(int j=0;j<layersSize[0];j++){
				weights[0][j]=new float[inputNumber];
				for(int k=0;k<inputNumber+1;k++){
					weights[0][j][k]=in.readFloat();
				}
			}
			
			if(layersNumber>1){
				for(byte i=1;i<layersNumber;i++){
					weights[i]=new float[layersSize[i]][];
					
					for(int j=0;j<layersSize[i];j++){
						weights[i][j]=new float[layersSize[i-1]+1];
						
						for(int k=0;k<weights[i][j].length;k++){
							weights[i][j][k]=in.readFloat();
						}
					}
				}
			}
			network.setWeights(weights);
			network.setFunction(function);
			break;
		default:in.close();throw new FileVersionException("Don't support file verion newer then -127. This file version: "+version);
		}
		in.close();
			
		return network;
	}
	public static void saveLNetwork(String fileName,LNetwork network) throws IOException{
		if(!network.isLearnning()) {
			File file=new File(fileName+".NN");
			file.createNewFile();
			DataOutputStream save=new DataOutputStream(new FileOutputStream(file));
			
			save.writeByte(-127);											//version of .NN
			
			save.writeByte(network.getFunction().getFunctionID());
			
			float[][][] weights=network.getWeights();
			save.writeInt(weights.length);
			
			save.writeInt(network.getInputNumber());
			
			for(int i=0;i<weights.length;i++){
				save.writeInt(weights[i].length);
			}
			for(int i=0;i<weights.length;i++){
				for(int j=0;j<weights[i].length;j++){
					for(int k=0;k<weights[i][j].length;k++){
						save.writeFloat(weights[i][j][k]);
					}
				}
			}
			save.close();
		}
		else throw new NeuralException(3);
	}
}