package org.ml4j.nn.factories;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AveragePoolingAxons;
import org.ml4j.nn.axons.AxonWeightsImpl;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.DefaultAveragePoolingAxonsImpl;
import org.ml4j.nn.axons.DefaultConvolutionalAxonsImpl;
import org.ml4j.nn.axons.DefaultFullyConnectedAxonsImpl;
import org.ml4j.nn.axons.DefaultMaxPoolingAxonsImpl;
import org.ml4j.nn.axons.DefaultScaleAndShiftAxonsImpl;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.MaxPoolingAxons;
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
		return new DefaultFullyConnectedAxonsImpl(leftNeurons, rightNeurons, new AxonWeightsImpl(leftNeurons.getNeuronCountExcludingBias(), 
				rightNeurons.getNeuronCountExcludingBias(), connectionWeights, biases, null));
	}

	@Override
	public FullyConnectedAxons createFullyConnectedAxons(Neurons leftNeurons, Neurons rightNeurons,
			Matrix connectionWeights, Matrix leftToRightBiases, Matrix rightToLeftBiases) {
		return new DefaultFullyConnectedAxonsImpl(leftNeurons, rightNeurons,  new AxonWeightsImpl(leftNeurons.getNeuronCountExcludingBias(), 
				rightNeurons.getNeuronCountExcludingBias(), connectionWeights, null, null));
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
		return new DefaultMaxPoolingAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config);
	}

	@Override
	public <N extends Neurons> ScaleAndShiftAxons<N> createScaleAndShiftAxons(N leftNeurons, N rightNeurons,
			Matrix gamma, Matrix beta) {
		return new DefaultScaleAndShiftAxonsImpl<>(matrixFactory, leftNeurons, rightNeurons);
	}

}
