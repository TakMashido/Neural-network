package liblaries.neuralNetwork;

import java.util.Scanner;

import liblaries.neuralNetwork.functions.FunctionList;
import liblaries.neuralNetwork.functions.Linear;
import liblaries.neuralNetwork.functions.OutputFunction;
import liblaries.neuralNetwork.learning.GrowingNeuralGas;
import liblaries.neuralNetwork.learning.HebbianNetwork;
import liblaries.neuralNetwork.learning.KohonenNetwork;
import liblaries.neuralNetwork.learning.LNetwork;

public class Others {
	public static final String version="1.0.3";
	
	public static LNetwork createLNetwork(String args) {
		return createLNetwork(args,false);
	}
	public static LNetwork createLNetwork(String args,boolean printMessages) {
		Scanner scanner=new Scanner(args);
		try {
		LNetwork network=null;
		
		if(!scanner.hasNext()) {
			if(printMessages)System.out.println("Please specifi type of network");
			return null;
		}
		String networkType=scanner.next();
		
		int[] layers=new int[] {150};
		int inputsNumber = 0;
		OutputFunction function=new Linear();
		
		StringBuffer otherCommands=new StringBuffer();
		while(scanner.hasNext()) {
			String command=scanner.next();
			switch(command) {
			case "-l":
				layers=new int[scanner.nextInt()];
				for(int i=0;i<layers.length;i++) {
					layers[i]=scanner.nextInt();
				}
				break;
			case "-i":
				inputsNumber=scanner.nextInt();
				break;
			case "-f":
				String data=scanner.next();
				function=FunctionList.geFunction(data);
				if(function==null) {
					if(printMessages)System.out.println(data+" function is not supported");
					return null;
				}
				break;
			default:
				otherCommands.append(command).append(" ");
			}
		}
		
		scanner.close();
		scanner=new Scanner(otherCommands.toString());
		
		if(inputsNumber==0) {
			if(printMessages)System.out.println("Please initialize inputs number. -i option");
		}
		
		switch(networkType) {
		case "kohonen":
			KohonenNetwork kohNetwork=new KohonenNetwork(inputsNumber,layers,function);
			network=kohNetwork;
			while(scanner.hasNext()) {
				String data=scanner.next();
				switch(data) {
				case "-n":					//neighborhood
					data=scanner.next();
					switch(data) {
					case "1d":
						kohNetwork.setLiner1DFunction();
						break;
					case "2d":
						data=scanner.next();
						if(data.equals("auto"))
							kohNetwork.setLinear2DSquereFunction((int)Math.sqrt(layers[0]));
						else
							kohNetwork.setLinear2DSquereFunction(Integer.parseInt(data));
						break;
					case "2deuculidean":
						data=scanner.next();
						if(data.equals("auto"))
							kohNetwork.setLinear2DEuclideanFunction((int)Math.sqrt(inputsNumber));
						else
							kohNetwork.setLinear2DEuclideanFunction(Integer.parseInt(data));
						break;
					default:
						if(printMessages)System.out.println(data+" neighborhood is not supported");
						return null;
					}
					break;
				case "-d":					//distance max min cyclesPerChange
					kohNetwork.setDistance(scanner.nextInt(), scanner.nextInt(), scanner.nextInt());
					break;
				default:
					if(printMessages)System.out.println(data+" is not supported option");
				}
			}
			break;
		case "hebbian":														//TODO end
			if(printMessages)System.out.println("Hebbian network is not fully supported yet");
			network=new HebbianNetwork(inputsNumber,layers,function);
			break;
		case "gng":
			int maxNeurons=200;
			int cyclesToAdd=15;
			int maxLife=500000;
			
			while(scanner.hasNext()) {
				String data=scanner.next();
				switch(data) {
				case "-mn":
					maxNeurons=scanner.nextInt();
					break;
				case "-ml":
					maxLife=scanner.nextInt();
					break;
				case "-ac":
					cyclesToAdd=scanner.nextInt();
					break;
				default:
					if(printMessages)System.out.println(data+" is not supported option");
				}
			}
			
			network=new GrowingNeuralGas(inputsNumber,maxNeurons,maxLife,cyclesToAdd);
			break;
		default:
			if(printMessages)System.out.println("Unsupported network type");
			return null;
		}
		
		return network;
		} finally {
			scanner.close();
		}
	}
}