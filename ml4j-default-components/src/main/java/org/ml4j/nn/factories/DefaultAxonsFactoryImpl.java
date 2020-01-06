package org.ml4j.nn.factories;

import java.util.Optional;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AveragePoolingAxons;
import org.ml4j.nn.axons.AxonWeightsInitialiser;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.DefaultAveragePoolingAxonsImpl;
import org.ml4j.nn.axons.DefaultConvolutionalAxonsImpl;
import org.ml4j.nn.axons.DefaultFullyConnectedAxonWeightsInitialiser;
import org.ml4j.nn.axons.DefaultFullyConnectedAxonsImpl;
import org.ml4j.nn.axons.DefaultMaxPoolingAxonsImpl;
import org.ml4j.nn.axons.DefaultScaleAndShiftAxonWeightsInitialiser;
import org.ml4j.nn.axons.DefaultScaleAndShiftAxonsImpl;
import org.ml4j.nn.axons.FullyConnectedAxonWeightsImpl;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.MaxPoolingAxons;
import org.ml4j.nn.axons.ScaleAndShiftAxonWeightsImpl;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public class DefaultAxonsFactoryImpl implements AxonsFactory {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private MatrixFactory matrixFactory;

	public DefaultAxonsFactoryImpl(MatrixFactory matrixFactory) {
		this.matrixFactory = matrixFactory;
	}

	@Override
	public FullyConnectedAxons createFullyConnectedAxons(Neurons leftNeurons, Neurons rightNeurons,
			Matrix connectionWeights, Matrix biases) {
		return createFullyConnectedAxons(leftNeurons, rightNeurons, connectionWeights, biases, null);
	}

	@Override
	public FullyConnectedAxons createFullyConnectedAxons(Neurons leftNeurons, Neurons rightNeurons,
			Matrix connectionWeights, Matrix leftToRightBiases, Matrix rightToLeftBiases) {

		AxonWeightsInitialiser axonWeightsInitialiser = new DefaultFullyConnectedAxonWeightsInitialiser(leftNeurons,
				rightNeurons);

		Matrix initialConnectionWeights = connectionWeights == null
				? axonWeightsInitialiser.getInitialConnectionWeights(matrixFactory)
				: connectionWeights;
		Optional<Matrix> initialLeftToRightBiases = leftToRightBiases == null
				? axonWeightsInitialiser.getInitialLeftToRightBiases(matrixFactory)
				: Optional.of(leftToRightBiases);
		Optional<Matrix> initialRightToLeftBiases = rightToLeftBiases == null
				? axonWeightsInitialiser.getInitialRightToLeftBiases(matrixFactory)
				: Optional.of(rightToLeftBiases);

		return new DefaultFullyConnectedAxonsImpl(leftNeurons, rightNeurons,
				new FullyConnectedAxonWeightsImpl(leftNeurons.getNeuronCountExcludingBias(),
						rightNeurons.getNeuronCountExcludingBias(), initialConnectionWeights,
						initialLeftToRightBiases.isPresent() ? initialLeftToRightBiases.get() : null,
						initialRightToLeftBiases.isPresent() ? initialRightToLeftBiases.get() : null));
	}

	@Override
	public AveragePoolingAxons createAveragePoolingAxons(Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig config) {
		return new DefaultAveragePoolingAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config);
	}

	@Override
	public ConvolutionalAxons createConvolutionalAxons(Neurons3D leftNeurons, Neurons3D rightNeurons,Axons3DConfig config, Matrix connectionWeights, Matrix biases) {
		return new DefaultConvolutionalAxonsImpl(this, leftNeurons, rightNeurons, config, connectionWeights, biases);
	}

	@Override
	public MaxPoolingAxons createMaxPoolingAxons(Neurons3D leftNeurons, Neurons3D rightNeurons, boolean scaleOutputs,
			Axons3DConfig config) {
		return new DefaultMaxPoolingAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config, scaleOutputs);
	}

	@Override
	public <N extends Neurons> ScaleAndShiftAxons<N> createScaleAndShiftAxons(N leftNeurons, N rightNeurons,
			Matrix gamma, Matrix beta) {

		AxonWeightsInitialiser axonWeightsInitialiser = new DefaultScaleAndShiftAxonWeightsInitialiser(leftNeurons);
		
		Matrix initialGamma = gamma == null
				? axonWeightsInitialiser.getInitialConnectionWeights(matrixFactory)
				: gamma;
		Optional<Matrix> initialBeta = beta == null
				? axonWeightsInitialiser.getInitialLeftToRightBiases(matrixFactory)
				: Optional.of(beta);
		
		ScaleAndShiftAxonWeightsImpl weights = new ScaleAndShiftAxonWeightsImpl(leftNeurons.getNeuronCountExcludingBias(), 
				rightNeurons.getNeuronCountExcludingBias(),initialGamma, initialBeta.isPresent() ? initialBeta.get() : null, null);
		return new DefaultScaleAndShiftAxonsImpl<>(leftNeurons, rightNeurons, weights);
	}

}
