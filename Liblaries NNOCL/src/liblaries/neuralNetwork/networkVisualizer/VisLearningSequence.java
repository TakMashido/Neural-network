package liblaries.neuralNetwork.networkVisualizer;

import java.awt.Color;
import java.awt.Graphics2D;
import java.util.Random;

import liblaries.neuralNetwork.learning.LearningSequence;

public class VisLearningSequence implements Visualizable{
	private LearningSequence[] ls;
	public VisLearningSequence(){}
	public VisLearningSequence(Point[] points, float radix, int number) {
		ls=new LearningSequence[points.length*number];
		
		Random random=new Random();
		
		int index=0;
		for(Point point:points) {
			for(int i=0;i<number;i++) {
				//(random.nextBoolean()?1:-1)
				float lenght=random.nextFloat()*radix;
				float direction=random.nextFloat();
				
				float x=(float)(lenght*Math.cos(Math.PI*direction))*(random.nextBoolean()?1:-1)+point.x;
				float y=(float)(lenght*Math.sin(Math.PI*direction))*(random.nextBoolean()?1:-1)+point.y;
				
				ls[index++]=new LearningSequence(new float[] {x,y});
			}
		}
	}
	public VisLearningSequence(float[][] inputs){
		if(inputs.length!=2)throw new VisException("Supports only 2 dimesions visualizable data");
		ls=LearningSequence.create(inputs);
	}
	
	public LearningSequence[] getLS() {
		return ls;
	}
	
	public void mixLS() {
		Random random=new Random();
		int ilEl=ls.length;
		LearningSequence[] newLS=new LearningSequence[ilEl];
		boolean[] included=new boolean[ilEl];									//True if LS elemnent is already in newLS
		
		int index;
				
		for(LearningSequence cu:ls){
			while(true){
				index=random.nextInt(ilEl);
				if(!included[index]){
					newLS[index]=cu;
					included[index]=true;
					break;
				}
			}
		}
		
		ls=newLS;
	}
	
	public void draw(Graphics2D g2d, float scale, float xOffset, float yOffset) {
		g2d.setColor(Color.BLACK);
		for(LearningSequence lsEl:ls) {
			g2d.fillRect(translate(lsEl.inputs[0],scale,xOffset), translate(lsEl.inputs[1],scale,yOffset), 1, 1);
		}
	}
	
	public static class Point{
		float x;
		float y;
		
		public Point() {
			x=y=0;
		}
		public Point(float x, float y) {
			this.x=x;
			this.y=y;
		}
	}
}