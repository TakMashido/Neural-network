package liblaries.neuralNetwork.networkVisualizer;

import java.awt.Graphics2D;

public interface Visualizable {
	public void draw(Graphics2D g2d, float scale, float xOffset, float yOffset);
	
	public default int translate(float value, float scale, float offset) {
		return (int)(value/scale-offset);
	}
}