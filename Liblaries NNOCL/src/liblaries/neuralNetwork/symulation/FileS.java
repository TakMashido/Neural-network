package liblaries.neuralNetwork.symulation;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

import liblaries.neuralNetwork.errors.FileVersionException;
import liblaries.neuralNetwork.functions.Function;
import liblaries.neuralNetwork.functions.FunctionList;

public class FileS {
	private static final byte NNSupportedVersion=-127;
	
	public static Network readNetwork(String fileName) throws IOException{
		DataInputStream in=new DataInputStream(new FileInputStream(fileName+".NN"));
		
		Function function;
		int layersNumber;
		int inputNumber;
		int[] layersSize;
		float[][][] weights;
		
		byte version=in.readByte();
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
			in.close();
			return new Network(inputNumber,weights,function);
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
			in.close();
			return new Network(inputNumber,weights,function);
		default :
			in.close();
			throw new FileVersionException("Don't support file verion newer then -127. This file version: "+version);
		}
	}
	
	public static byte getNNSupportedVersion() {
		return NNSupportedVersion;
	}
}