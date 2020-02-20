package org.ml4j.nn.axons;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;
import org.ml4j.nn.neurons.format.features.FeaturesFormatImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FullyConnectedAxonWeightsImpl extends AxonWeightsBase implements AxonWeights {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(FullyConnectedAxonWeightsImpl.class);

	public FullyConnectedAxonWeightsImpl(int inputNeuronCount, int outputNeuronCount, WeightsMatrix connectionWeights,
			BiasVector leftToRightBiases, BiasVector rightToLeftBiases) {
		super(inputNeuronCount, outputNeuronCount, connectionWeights, leftToRightBiases, rightToLeftBiases,
				AxonWeightsType.FULLY_CONNECTED);
		if (connectionWeights != null && connectionWeights.getFormat().getOrientation() != WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS) {
			throw new IllegalArgumentException("Currently only " + WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS + " weights format supported");
		}
		if (leftToRightBiases != null && leftToRightBiases.getFormat().getOrientation() != FeaturesVectorOrientation.COLUMN_VECTOR) {
			throw new IllegalArgumentException("Currently only " + FeaturesVectorOrientation.COLUMN_VECTOR + " bias format supported");
		}
		if (rightToLeftBiases != null && rightToLeftBiases.getFormat().getOrientation() != FeaturesVectorOrientation.COLUMN_VECTOR) {
			throw new IllegalArgumentException("Currently only " + FeaturesVectorOrientation.COLUMN_VECTOR + " bias format supported");
		}
	}

	@Override
	public NeuronsActivation applyToRightToLeftInput(NeuronsActivation input, AxonsContext axonsContext) {
		
		if (input.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException("Only feature orientation:" + NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET + " currently supported");
		}
		
		Matrix axonWeightsInput = input.getActivations(axonsContext.getMatrixFactory());
		
		EditableMatrix output = connectionWeights.getMatrix().transpose().mmul(axonWeightsInput).asEditableMatrix();
		if (rightToLeftBiases != null) {
			output.addiColumnVector(rightToLeftBiases.getVector());
		}
		// TODO format
		return new NeuronsActivationImpl(new Neurons(this.inputNeuronCount, false), output, input.getFormat());
	}

	@Override
	public NeuronsActivation applyToLeftToRightInput(NeuronsActivation input, AxonsContext axonsContext) {
				
		if (input.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException("Only feature orientation:" + NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET + " currently supported");
		}
		
		List<Dimension> inputActivationDimensions = input.getFeatureOrientation() == 
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET ? input.getFormat().getRowDimensions() : input.getFormat().getColumnDimensions();
				
		List<Dimension> decomposedInputDimensions = connectionWeights.getFormat().getInputDimensions()
				.stream().flatMap(d -> d.decompose().stream()).collect(Collectors.toList());
				
		if (!Dimension.isEquivalent(decomposedInputDimensions, 
				Arrays.asList(Dimension.INPUT_FEATURE), DimensionScope.INPUT) && !Dimension.isEquivalent(
						decomposedInputDimensions, inputActivationDimensions, DimensionScope.INPUT))	{
			LOGGER.error("Activation input dimensions:" + inputActivationDimensions);
			LOGGER.error("Weights input dimensions:" + decomposedInputDimensions);
			throw new IllegalArgumentException("Input/Weight Dimensions don't match");
		}
		
		List<Dimension> inputExampleDimensions = input.getFeatureOrientation() == 
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET ? input.getFormat().getColumnDimensions() : input.getFormat().getRowDimensions();
				
		List<Dimension> allOutputDimensions = new ArrayList<>();
		allOutputDimensions.addAll(connectionWeights.getFormat().getOutputDimensions());
		allOutputDimensions.addAll(connectionWeights.getFormat().getInputDimensions());
		
		NeuronsActivationFormat<?> outputFormat = new NeuronsActivationFormat<>(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, 
				new FeaturesFormatImpl(connectionWeights.getFormat().getOutputDimensions()), inputExampleDimensions);
	
		Matrix axonWeightsInput = input.getActivations(axonsContext.getMatrixFactory());
		
		EditableMatrix output = connectionWeights.getMatrix().mmul(axonWeightsInput).asEditableMatrix();
		if (leftToRightBiases != null) {
			output.addiColumnVector(leftToRightBiases.getVector());
		}
		return new NeuronsActivationImpl(new Neurons(this.outputNeuronCount, false), output, outputFormat);
	}

	@Override
	public AxonWeights dup() {
		return new FullyConnectedAxonWeightsImpl(inputNeuronCount, outputNeuronCount, connectionWeights.dup(),
				leftToRightBiases == null ? null : leftToRightBiases.dup(),
				rightToLeftBiases == null ? null : rightToLeftBiases.dup());
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return Dimension.isEquivalent(format.getFeaturesFormat().getDimensions(), connectionWeights.getFormat().getInputDimensions(), DimensionScope.INPUT);
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}

}
