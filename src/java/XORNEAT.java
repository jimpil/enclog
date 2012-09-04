
import org.encog.engine.network.activation.ActivationStep;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.training.NEATTraining;
import org.encog.neural.networks.training.CalculateScore;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.util.simple.EncogUtility;

/**
 * XOR-NEAT: This example solves the classic XOR operator neural
 * network problem.  However, it uses a NEAT evolving network.
 * 
 * @author $Author$
 * @version $Revision$
 */
public class XORNEAT {
        public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
                        { 0.0, 1.0 }, { 1.0, 1.0 } };

        public static double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };

        public static void main(final String args[]) {

                MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
                NEATPopulation pop = new NEATPopulation(2,1,1000);
                CalculateScore score = new TrainingSetScore(trainingSet);
                // train the neural network
                ActivationStep step = new ActivationStep();
                step.setCenter(0.5);
                pop.setNeatActivationFunction(step);
                
                final NEATTraining train = new NEATTraining(score,pop);
                
                EncogUtility.trainToError(train, 0.01);

                NEATNetwork network = (NEATNetwork)train.getMethod();

                //network.clearContext();
                // test the neural network
                System.out.println("Neural Network Results:");
                EncogUtility.evaluate(network, trainingSet);
        }
}
