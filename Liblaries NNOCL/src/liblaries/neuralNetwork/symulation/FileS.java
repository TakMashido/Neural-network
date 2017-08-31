package liblaries.neuralNetwork.symulation;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

import liblaries.neuralNetwork.errors.FileVersionException;
import liblaries.neuralNetwork.functions.Function;
import liblaries.neuralNetwork.functions.FunctionList;

public class FileS {
	public static Network readNetwork(String Nazwa) throws IOException{
		DataInputStream in=new DataInputStream(new FileInputStream(Nazwa+".SN"));
		
		float[][][] weights;
		byte version=in.readByte();
		switch(version){
		case -128:
			Function function=FunctionList.getFunction(in.readByte());
			
			byte layersNumber=in.readByte();
			int inputNumber=in.readInt();
			
			int[] layersSize=new int[layersNumber];
			
			weights=new float[layersNumber][][];
			for(byte i=0;i<layersNumber;i++){
				layersSize[i]=in.readInt();
				weights[i]=new float[layersSize[i]][];
			}
			
			for(int j=0;j<layersSize[0];j++){
				weights[0][j]=new float[inputNumber];
				for(int k=0;k<inputNumber;k++){
					weights[0][j][k]=(float) in.readDouble();
				}
			}
			
			if(layersNumber>1){
				for(byte i=1;i<layersNumber;i++){
					weights[i]=new float[layersSize[i]][];
					
					for(int j=0;j<layersSize[i];j++){
						weights[i][j]=new float[layersSize[i-1]];
						
						for(int k=0;k<layersSize[i-1];k++){
							weights[i][j][k]=(float) in.readDouble();
						}
					}
				}
			}
			in.close();
			return new Network(inputNumber,weights,function);
		default :
			in.close();
			throw new FileVersionException("Don't support file verion newer then -128. This file version: "+version);
		}
	}
}