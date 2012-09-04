import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

public class NeuralPilot {
        
        private BasicNetwork network;
        private boolean track;
    private NormalizedField fuelStats;
    private NormalizedField altitudeStats;
    private NormalizedField velocityStats;
        
        public NeuralPilot(BasicNetwork network, boolean track)
        {
        fuelStats = new NormalizedField(NormalizationAction.Normalize, "fuel", 200, 0, -0.9, 0.9);
        altitudeStats = new NormalizedField(NormalizationAction.Normalize, "altitude", 10000, 0, -0.9, 0.9);
        velocityStats = new NormalizedField(NormalizationAction.Normalize, "velocity", LanderSimulator.TERMINAL_VELOCITY, -LanderSimulator.TERMINAL_VELOCITY, -0.9, 0.9);

                this.track = track;
                this.network = network;
        }
        
        public int scorePilot()
        {
                LanderSimulator sim = new LanderSimulator();
                while(sim.flying())
                {
                        MLData input = new BasicMLData(3);
            input.setData(0, this.fuelStats.normalize(sim.getFuel()));
            input.setData(1, this.fuelStats.normalize(sim.getAltitude()));
            input.setData(2, this.fuelStats.normalize(sim.getVelocity()));
            MLData output = this.network.compute(input);
            double value = output.getData(0);

            boolean thrust;
                        
                        if( value > 0 )
                        {
                                thrust = true;
                                if( track )
                                        System.out.println("THRUST");
                        }
                        else
                                thrust = false;
                        
                        sim.turn(thrust);
                        if( track )
                                System.out.println(sim.telemetry());
                }
                return(sim.score());
        }
}
