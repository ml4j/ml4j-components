/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.factories;

import java.util.Arrays;
import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.BiasMatrixImpl;
import org.ml4j.nn.axons.PassThroughAxonsImpl;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.activationfunctions.DefaultDifferentiableActivationFunctionComponentImpl;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.axons.BatchNormDirectedAxonsComponent;
import org.ml4j.nn.components.axons.DefaultBatchNormDirectedAxonsComponentImpl;
import org.ml4j.nn.components.axons.DefaultDirectedAxonsComponentImpl;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.DefaultComponentBatchImpl;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentBatch;
import org.ml4j.nn.components.manytoone.DefaultManyToOneFilterConcatDirectedComponentImpl;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetomany.DefaultOneToManyDirectedComponentImpl;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.SerializableIntSupplier;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetoone.DefaultDirectedComponentBipoleGraphImpl;
import org.ml4j.nn.components.onetoone.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.components.onetoone.DefaultSpaceToDepthDirectedComponent;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;

/**
 * Default implementation of DirectedComponentFactory
 * 
 * @author Michael Lavelle
 */
public class DefaultDirectedComponentFactoryImpl implements DirectedComponentFactory {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	private MatrixFactory matrixFactory;
	private AxonsFactory axonsFactory;
	private DifferentiableActivationFunctionFactory activationFunctionFactory;
	private DirectedComponentFactory directedComponentFactory;

	public DefaultDirectedComponentFactoryImpl(MatrixFactory matrixFactory, AxonsFactory axonsFactory,
			DifferentiableActivationFunctionFactory activationFunctionFactory) {
		this.matrixFactory = matrixFactory;
		this.axonsFactory = axonsFactory;
		this.directedComponentFactory = this;
		this.activationFunctionFactory = activationFunctionFactory;
	}

	public void setDirectedComponentFactory(DirectedComponentFactory directedComponentFactory) {
		this.directedComponentFactory = directedComponentFactory;
	}

	@Override
	public DirectedAxonsComponent<Neurons, Neurons, ?> createFullyConnectedAxonsComponent(String name, Neurons leftNeurons,
			Neurons rightNeurons, WeightsMatrix connectionWeights, BiasMatrix biases) {
		return createDirectedAxonsComponent(name,
				axonsFactory.createFullyConnectedAxons(leftNeurons, rightNeurons, 
						connectionWeights, 
						biases));
	}

	@Override
	public <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R, ?> createDirectedAxonsComponent(String name, 
			Axons<L, R, ?> axons) {
		return new DefaultDirectedAxonsComponentImpl<>(name, axons);
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createConvolutionalAxonsComponent(String name, Neurons3D leftNeurons,
			Neurons3D rightNeurons, Axons3DConfig config, WeightsMatrix connectionWeights, BiasMatrix biases) {
		return createDirectedAxonsComponent(name,
				axonsFactory.createConvolutionalAxons(leftNeurons, rightNeurons, config, connectionWeights, biases));
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createMaxPoolingAxonsComponent(String name, Neurons3D leftNeurons,
			Neurons3D rightNeurons, Axons3DConfig config, boolean scaleOutputs) {
		return createDirectedAxonsComponent(name,
				axonsFactory.createMaxPoolingAxons(leftNeurons, rightNeurons, scaleOutputs, config));
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createAveragePoolingAxonsComponent(String name, Neurons3D leftNeurons,
			Neurons3D rightNeurons, Axons3DConfig config) {
		return createDirectedAxonsComponent(name, axonsFactory.createAveragePoolingAxons(leftNeurons, rightNeurons, config));
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, ?> createBatchNormAxonsComponent(String name, N leftNeurons,
			N rightNeurons) {
		throw new UnsupportedOperationException();
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, ?> createBatchNormAxonsComponent(String name, N leftNeurons,
			N rightNeurons, WeightsMatrix gamma, BiasMatrix beta, Matrix mean, Matrix stddev) {
		throw new UnsupportedOperationException();
	}

	@Override
	public BatchNormDirectedAxonsComponent<Neurons3D, ?> createConvolutionalBatchNormAxonsComponent(String name, 
			Neurons3D leftNeurons, Neurons3D rightNeurons) {
		return new DefaultBatchNormDirectedAxonsComponentImpl<>(name, 
				axonsFactory.createScaleAndShiftAxons(leftNeurons, rightNeurons, 
						new WeightsMatrixImpl(null, 
								new WeightsFormatImpl(Arrays.asList(
										Dimension.INPUT_DEPTH, 
										Dimension.INPUT_HEIGHT, 
										Dimension.INPUT_WIDTH), 
										Arrays.asList(Dimension.OUTPUT_FEATURE),
										WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)), null), null, null, true);
	}

	@Override
	public BatchNormDirectedAxonsComponent<Neurons3D, ?> createConvolutionalBatchNormAxonsComponent(String name, 
			Neurons3D leftNeurons, Neurons3D rightNeurons, WeightsMatrix gamma, BiasMatrix beta, Matrix mean, Matrix stddev) {
		return new DefaultBatchNormDirectedAxonsComponentImpl<>(name, 
				axonsFactory.createScaleAndShiftAxons(leftNeurons, rightNeurons,
						new WeightsMatrixImpl(expandChannelValuesToFeatureValues(matrixFactory, rightNeurons, gamma),
								new WeightsFormatImpl(Arrays.asList(
										Dimension.INPUT_DEPTH, 
										Dimension.INPUT_HEIGHT, 
										Dimension.INPUT_WIDTH), 
										Arrays.asList(Dimension.OUTPUT_FEATURE),
										WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)),
						beta == null ? null : new BiasMatrixImpl(expandChannelValuesToFeatureValues(matrixFactory, rightNeurons, beta))),
				expandChannelValuesToFeatureValues(matrixFactory, rightNeurons, mean),
				expandChannelValuesToFeatureValues(matrixFactory, rightNeurons, stddev), true);
	}
	
	public static Matrix expandChannelValuesToFeatureValues(MatrixFactory matrixFactory, Neurons3D rightNeurons,
			WeightsMatrix channelValues) {
		if (channelValues == null) {
			return null;
		}
		if (channelValues != null && !Dimension.isEquivalent(channelValues.getFormat().getInputDimensions(), Arrays.asList(Dimension.INPUT_DEPTH), DimensionScope.INPUT)) {
			throw new IllegalArgumentException("Expected batch norm to be of format with input dimensions:" +  Dimension.INPUT_DEPTH);
		}
		return expandChannelValuesToFeatureValues(matrixFactory, rightNeurons, channelValues.getWeights());
	}

	public static Matrix expandChannelValuesToFeatureValues(MatrixFactory matrixFactory, Neurons3D rightNeurons,
			Matrix channelValues) {
		if (channelValues == null) {
			return null;
		}
	
		float[] channelValuesArray = channelValues.getRowByRowArray();
		float[] channelValuesExpanded = new float[rightNeurons.getNeuronCountExcludingBias()];
		int index = 0;
		for (int channel = 0; channel < channelValuesArray.length; channel++) {
			for (int i = 0; i < rightNeurons.getWidth() * rightNeurons.getHeight(); i++) {
				channelValuesExpanded[index++] = channelValuesArray[channel];
			}
		}
		return matrixFactory.createMatrixFromRowsByRowsArray(channelValuesExpanded.length, 1, channelValuesExpanded);
	}

	@Override
	public <N extends Neurons> DirectedAxonsComponent<N, N, ?> createPassThroughAxonsComponent(String name, N leftNeurons,
			N rightNeurons) {
		return createDirectedAxonsComponent(name, new PassThroughAxonsImpl<>(leftNeurons, rightNeurons));
	}

	@Override
	public OneToManyDirectedComponent<?> createOneToManyDirectedComponent(SerializableIntSupplier targetComponentsCount) {
		return new DefaultOneToManyDirectedComponentImpl(targetComponentsCount);
	}

	@Override
	public ManyToOneDirectedComponent<?> createManyToOneDirectedComponent(Neurons outputNeurons,
			PathCombinationStrategy pathCombinationStrategy) {
		// TODO
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public ManyToOneDirectedComponent<?> createManyToOneDirectedComponent3D(Neurons3D outputNeurons,
			PathCombinationStrategy pathCombinationStrategy) {
		if (pathCombinationStrategy == PathCombinationStrategy.FILTER_CONCAT) {
			return new DefaultManyToOneFilterConcatDirectedComponentImpl(outputNeurons);
		} else {
			// TODO
			throw new UnsupportedOperationException("Not yet implemented");
		}
	}

	@Override
	public DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponent(String name, Neurons neurons,
			DifferentiableActivationFunction differentiableActivationFunction) {
		return new DefaultDifferentiableActivationFunctionComponentImpl(name, neurons, differentiableActivationFunction);
	}

	@Override
	public DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponent(String name, Neurons neurons,
			ActivationFunctionType activationFunctionType, ActivationFunctionProperties activationFunctionProperties) {
		DifferentiableActivationFunction differentiableActivationFunction = activationFunctionFactory
				.createActivationFunction(activationFunctionType, activationFunctionProperties);
		return new DefaultDifferentiableActivationFunctionComponentImpl(name, neurons, differentiableActivationFunction);
	}

	@Override
	public DefaultDirectedComponentBipoleGraph createDirectedComponentBipoleGraph(String name, Neurons leftNeurons,
			Neurons rightNeurons, List<DefaultChainableDirectedComponent<?, ?>> parallelComponentBatch,
			PathCombinationStrategy pathCombinationStrategy) {

		if (rightNeurons instanceof Neurons3D) {
			return new DefaultDirectedComponentBipoleGraphImpl(name, directedComponentFactory, leftNeurons,
					(Neurons3D) rightNeurons, createDirectedComponentBatch(parallelComponentBatch),
					pathCombinationStrategy);
		} else {
			return new DefaultDirectedComponentBipoleGraphImpl(name, directedComponentFactory, leftNeurons, rightNeurons,
					createDirectedComponentBatch(parallelComponentBatch), pathCombinationStrategy);
		}
	}

	@Override
	public DefaultDirectedComponentChain createDirectedComponentChain(
			List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents) {
		return new DefaultDirectedComponentChainImpl(sequentialComponents);
	}

	// @Override
	public DefaultDirectedComponentBatch createDirectedComponentBatch(
			List<DefaultChainableDirectedComponent<?, ?>> parallelComponents) {
		return new DefaultComponentBatchImpl(parallelComponents);
	}

	@Override
	public DefaultChainableDirectedComponent<?, ?> createComponent(String name, 
			Neurons leftNeurons, Neurons rightNeurons, NeuralComponentType neuralComponentType) {
		if ("SPACE_TO_DEPTH".equals(neuralComponentType.getId()) && leftNeurons instanceof Neurons3D && rightNeurons instanceof Neurons3D) {
			Neurons3D left = (Neurons3D)leftNeurons;
			Neurons3D right = (Neurons3D)rightNeurons;
			int blockHeight = left.getHeight() / right.getHeight();
			int blockWidth = left.getWidth() / right.getWidth();
			return new DefaultSpaceToDepthDirectedComponent(name, left, right, blockHeight, blockWidth);
		}
		throw new UnsupportedOperationException("Creation of component by component type not yet implemented");
	}
}
