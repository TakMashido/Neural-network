package liblaries.neuralNetwork.networkVisualizer;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Random;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextField;

import liblaries.neuralNetwork.Others;
import liblaries.neuralNetwork.learning.KohonenNetwork;
import liblaries.neuralNetwork.learning.LNetwork;
import liblaries.neuralNetwork.networkVisualizer.VisLearningSequence.Point;
import liblaries.neuralNetwork.networkVisualizer.networks.VisKohonen;
import liblaries.neuralNetwork.networkVisualizer.networks.VisNetwork;

public class Visualizer extends JFrame implements ActionListener{
	private static final long serialVersionUID = 1L;
	
	private VisTeacher teacher;
	private VisNetwork network;
	private VisLearningSequence ls;
	
	private float scale=1;
	private float xOffset=0;
	private float yOffset=0;
	
	private Runner runner;
	private boolean simulate=false;
	private long delay=2000;
	
	private NetworkPanel panel;
	private JTextField txtDelay;
	private JTextField txtScale;
	private JTextField txtXOffset;
	private JTextField txtYOffset;
	private JButton btnTick;
	private JButton btnBigTick;
	private JButton btnStart;
	private JButton btnStop;
	
	public Visualizer(VisNetwork visNetwork, VisTeacher teacher, VisLearningSequence visLS) {
		xOffset=-125;
		yOffset=-125;
		scale=0.05f;
		visNetwork.getNetwork().setLS(visLS.getLS());
		network=visNetwork;
		this.teacher=teacher;
		teacher.setNetwork(visNetwork.getNetwork());
		ls=visLS;
		
		setTitle("Network visualizer");
		
		setLayout(new BorderLayout(0, 0));
		
		panel = new NetworkPanel();
		add(panel, BorderLayout.CENTER);
		
		JPanel miscPanel = new JPanel();
		add(miscPanel, BorderLayout.SOUTH);
		miscPanel.setLayout(new GridLayout(3, 0, 0, 0));
		
		btnTick=new JButton("Tick");
		btnTick.addActionListener(this);
		miscPanel.add(btnTick);
		
		miscPanel.add(new JPanel());
		
		btnBigTick=new JButton("Big tick");
		btnBigTick.addActionListener(this);
		miscPanel.add(btnBigTick);
		
		btnStart = new JButton("Start");
		btnStart.addActionListener(this);
		miscPanel.add(btnStart);
		
		txtDelay = new JTextField("Delay");
		txtDelay.addActionListener(this);
		miscPanel.add(txtDelay);
		
		btnStop = new JButton("Stop");
		btnStop.addActionListener(this);
		miscPanel.add(btnStop);
		
		txtScale=new JTextField("scale");
		txtScale.addActionListener(this);
		miscPanel.add(txtScale);
		
		txtXOffset=new JTextField("x offset");
		txtXOffset.addActionListener(this);
		miscPanel.add(txtXOffset);
		
		txtYOffset=new JTextField("y offset");
		txtYOffset.addActionListener(this);
		miscPanel.add(txtYOffset);
		
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setSize(300,300);
		setVisible(true);
	}
	
	public void actionPerformed(ActionEvent e) {
		Object source=e.getSource();
		if(source==btnTick) {
			tick();
		}else if(source==btnBigTick) {
			bigTick();
		}else if(source==btnStart) {
			runner=new Runner();
			runner.setDaemon(true);
			runner.start();
		}else if(source==txtDelay) {
			try {
				delay=Integer.parseInt(txtDelay.getText());
			}catch(NumberFormatException ex) {
				System.out.println(txtDelay.getText()+" is not right number");
			}
		}else if(source==btnStop) {
			simulate=false;
		}else if(source==txtScale) {
			try {
				scale=Float.parseFloat(txtScale.getText());
				panel.repaint();
			}catch(NumberFormatException ex) {
				System.out.println(txtScale.getText()+" is not right number");
			}
		}else if(source==txtXOffset) {
			try {
				xOffset=Float.parseFloat(txtXOffset.getText());
				panel.repaint();
			}catch(NumberFormatException ex) {
				System.out.println(txtXOffset.getText()+" is not right number");
			}
		}else if(source==txtYOffset) {
			try {
				yOffset=Float.parseFloat(txtYOffset.getText());
				panel.repaint();
			}catch(NumberFormatException ex) {
				System.out.println(txtYOffset.getText()+" is not right number");
			}
		}
	}
	
	private void tick() {
		teacher.tick();
		
		panel.repaint();
	}
	public void bigTick() {
		teacher.bigTick();
		
		panel.repaint();
	}
	
	private class Runner extends Thread{
		public Runner() {
			Thread.currentThread().setName("Runner");
			simulate=true;
		}
		
		public void run() {
			while(simulate) {
				try {
					Thread.sleep(delay);
				} catch (InterruptedException e) {
					break;
				}
				tick();
			}
		}
	}
	private class NetworkPanel extends JPanel{
		private static final long serialVersionUID = 3325201809651557475L;

		protected void paintComponent(Graphics g) {
			Graphics2D g2d=(Graphics2D)g;
			
			g2d.setColor(Color.WHITE);
			g2d.fillRect(0, 0, getWidth(), getHeight());
			
			ls.draw(g2d, scale, xOffset, yOffset);
			network.draw(g2d, scale, xOffset, yOffset);
		}
	}
	
	public static void main(String[]args) {
		LNetwork network=Others.createLNetwork("kohonen -i 2 -l 1 400 -n 2d 1 -d 10 1 1",true);
		VisTeacher teacher=new VisTeacher();
		VisLearningSequence visLS = new VisLearningSequence(new Point[] {new Point(0,0),new Point(1,1),new Point(10,-1)},.5f,1000);
		visLS.mixLS();
		
		if(network instanceof KohonenNetwork)new Visualizer(VisNetwork.createNetwork(network),teacher,visLS);
		
	}
}