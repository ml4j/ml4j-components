package org.ml4j.nn.axons;

import java.util.Optional;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons;

public class DefaultFullyConnectedAxonWeightsInitialiser implements AxonWeightsInitialiser {
	
	private Neurons leftNeurons;
	private Neurons rightNeurons;

	public DefaultFullyConnectedAxonWeightsInitialiser(Neurons leftNeurons, Neurons rightNeurons) {
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
	}

	@Override
	public Matrix getInitialConnectionWeights(MatrixFactory matrixFactory) {
		return matrixFactory.createRandn(rightNeurons.getNeuronCountExcludingBias(), 
				leftNeurons.getNeuronCountExcludingBias()).asEditableMatrix().muli((float)Math.sqrt(1f / leftNeurons.getNeuronCountExcludingBias()));
	}

	@Override
	public Optional<Matrix> getInitialLeftToRightBiases(MatrixFactory matrixFactory) {
		if (leftNeurons.hasBiasUnit()) {
			Matrix randn = matrixFactory.createRandn(rightNeurons.getNeuronCountExcludingBias(), 1)
					.asEditableMatrix().muli((float)Math.sqrt(1f / leftNeurons.getNeuronCountExcludingBias()));
			return Optional.of(randn);
		} else {
			return Optional.empty();
		}
	}

	@Override
	public Optional<Matrix> getInitialRightToLeftBiases(MatrixFactory matrixFactory) {
		if (rightNeurons.hasBiasUnit()) {
			Matrix randn = matrixFactory.createRandn(rightNeurons.getNeuronCountExcludingBias(), 1)
					.asEditableMatrix().muli((float)Math.sqrt(1f / leftNeurons.getNeuronCountExcludingBias()));
			return Optional.of(randn);
		} else {
			return Optional.empty();
		}
	}
}
