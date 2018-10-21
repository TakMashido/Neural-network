package liblaries.neuralNetwork.networkVisualizer.networks;

import java.awt.Color;
import java.awt.Graphics2D;

import liblaries.neuralNetwork.learning.KohonenNetwork;
import liblaries.neuralNetwork.learning.LNetwork;

public class VisKohonen extends VisNetwork{
	private KohonenNetwork network;
	
	public VisKohonen(KohonenNetwork network) {
		this.network=network;
	}
	public void draw(Graphics2D g2d, float scale, float xOffset, float yOffset) {
		g2d.setColor(Color.BLUE.darker());
		
		int[] layersSize=network.getLayersSize();
		float[][][] weights=network.getWeights();
		int[] rowsNumber=network.getRowsNumber();
		
		int[] x=new int[weights[0].length];
		int[] y=new int[weights[0].length];
		for(int i=0;i<weights[0].length;i++) {
			x[i]=translate(weights[0][i][1],scale,xOffset);
			y[i]=translate(weights[0][i][2],scale,yOffset);
		}
		
		for(int i=0;i<layersSize[1];i++) {
			int index=i+1;
			if((i+1)%rowsNumber[0]!=0&index<layersSize[1]) {
				g2d.drawLine(x[i], y[i], x[index], y[index]);
			}
			index=i+rowsNumber[0];
			if(index<layersSize[1]) {
				g2d.drawLine(x[i], y[i], x[index], y[index]);
			}
		}
		
		g2d.setColor(Color.BLUE);
		for(int i=0;i<layersSize[1];i++) {
			g2d.fillOval(x[i]-2, y[i]-2, 5, 5);
		}
	}

	
	@SuppressWarnings("unchecked")
	public <T extends LNetwork> T getNetwork(T networkType) {
		return (T)network;
	}
	public LNetwork getNetwork() {
		return network;
	}
}