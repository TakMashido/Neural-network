package liblaries.neuralNetwork.learning;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import liblaries.neuralNetwork.errors.NeuralException;
import liblaries.neuralNetwork.functions.OutputFunction;
import liblaries.neuralNetwork.functions.FunctionList;

public class FileL {
	public static final byte NNSupportedVersion=-126;
	public static final byte LSSupportedVersion=-128;
	
	public static void saveLS(String fileName,LearningSequence[] data) throws IOException{
		DataOutputStream out=new DataOutputStream(new FileOutputStream(fileName+".LS"));
		
		out.writeByte(-128);							//File version
		
		out.writeInt(data.length);
		out.writeInt(data[0].inputs.length);
		out.writeInt(data[0].outputs.length);
		
		for(LearningSequence element:data){
			for(float input:element.inputs){
				out.writeDouble(input);
			}
			for(float wyjœcie:element.outputs){
				out.writeDouble(wyjœcie);
			}
		}
		
		out.close();
	}
	public static LearningSequence[] readLS(String fileName) throws IOException{
		DataInputStream in=new DataInputStream(new FileInputStream(fileName+".LS"));		
		
		byte version=in.readByte();
		switch(version){											//Version
		case -128:
			int elementNumber=in.readInt();
			int inputNumber=in.readInt();
			int outputNumber=in.readInt();
			
			LearningSequence[] Return=new LearningSequence[elementNumber];
			
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
		default:in.close();throw new IOException("Don't support file verion newer then -127. This file version: "+version);
		}
	}
	
	public static LNetwork readLNetwork(String fileName) throws IOException {
		return readLNetwork(fileName, new BackPropagationNetwork());
	}
	public static LNetwork readLNetwork(String fileName, LNetwork network) throws IOException{
		DataInputStream in=new DataInputStream(new FileInputStream(fileName+".NN"));
		
		OutputFunction outputFunction;
		int layersNumber;
		int inputNumber;
		int[] layersSize;
		float[][][] weights;
		
		byte version=in.readByte();											//file version
		switch(version){
		case -128:
			outputFunction=FunctionList.getFunction(in.readByte());
			in.readByte();													//add support of diffrend simulation options in LNetwork
			
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
			network.setFunction(outputFunction);
			break;
		case -127:
			outputFunction=FunctionList.getFunction(in.readByte());
			
			layersNumber=in.readInt();
			inputNumber=in.readInt()+1;
			
			layersSize=new int[layersNumber];
			
			weights=new float[layersNumber][][];
			for(byte i=0;i<layersNumber;i++){								//Size of each layer
				layersSize[i]=in.readInt();
				weights[i]=new float[layersSize[i]][];
			}
			
			for(int j=0;j<layersSize[0];j++){
				weights[0][j]=new float[inputNumber];
				for(int k=0;k<inputNumber;k++){
					weights[0][j][k]=in.readFloat();
				}
			}
			
			if(layersNumber>1){
				for(byte i=1;i<layersNumber;i++){					
					for(int j=0;j<layersSize[i];j++){
						weights[i][j]=new float[layersSize[i-1]+1];
						
						for(int k=0;k<weights[i][j].length;k++){
							weights[i][j][k]=in.readFloat();
						}
					}
				}
			}
			network.setWeights(weights);
			network.setFunction(outputFunction);
			break;
		default:in.close();throw new IOException("Don't support file verion newer then -127. This file version: "+version);
		}
		in.close();
			
		return network;
	}
	public static void saveLNetwork(String fileName,LNetwork network) throws IOException{
		if(!network.isLearnning()) {
			if(fileName.toLowerCase().endsWith(".nn"))
				fileName=fileName.substring(0, fileName.length()-3);
			File file=new File(fileName+".NN");
			file.createNewFile();
			DataOutputStream save=new DataOutputStream(new FileOutputStream(file));
			
			save.writeByte(-126);											//version of .NN
			
			save.writeByte(network.getFunction().getFunctionID());
			save.writeByte(network.getSimMethodID());
			
			float[][][] weights=network.getWeights();
			save.writeInt(network.layersNumber);
			
			for(int i=0;i<network.layersNumber+1;i++){
				save.writeInt(network.layersSize[i]);
			}
			for(int i=0;i<network.layersNumber;i++){
				for(int j=0;j<network.layersSize[i+1];j++){
					for(int k=0;k<network.layersSize[i];k++){
						save.writeFloat(weights[i][j][k]);
					}
				}
			}
			save.close();
		}
		else throw new NeuralException(3);
	}
}