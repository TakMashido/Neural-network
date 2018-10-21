package liblaries.neuralNetwork.networkVisualizer.networks;

import liblaries.neuralNetwork.learning.KohonenNetwork;
import liblaries.neuralNetwork.learning.LNetwork;
import liblaries.neuralNetwork.networkVisualizer.Visualizable;

public abstract class VisNetwork implements Visualizable{
	//private networkType network
	
	public static VisNetwork createNetwork(LNetwork origin) {
		if(origin instanceof KohonenNetwork)return new VisKohonen((KohonenNetwork)origin);
		return null;
	}
	
	public abstract <T extends LNetwork> T getNetwork(T networkType);
	public abstract LNetwork getNetwork();
}