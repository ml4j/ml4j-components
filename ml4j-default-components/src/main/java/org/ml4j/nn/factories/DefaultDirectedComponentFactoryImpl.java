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
import org.ml4j.nn.axons.AxonsConfig;
import org.ml4j.nn.axons.AxonsContextConfigurer;
import org.ml4j.nn.axons.AxonsType;
import org.ml4j.nn.axons.BatchNormAxonsConfig;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.BiasVectorImpl;
import org.ml4j.nn.axons.ConvolutionalAxonsConfig;
import org.ml4j.nn.axons.DefaultSpaceToDepthAxons;
import org.ml4j.nn.axons.FeaturesVector;
import org.ml4j.nn.axons.FeaturesVectorFormat;
import org.ml4j.nn.axons.FullyConnectedAxonsConfig;
import org.ml4j.nn.axons.GenericAxonsAdapter;
import org.ml4j.nn.axons.GenericTrainableAxonsAdapter;
import org.ml4j.nn.axons.PassThroughAxonsImpl;
import org.ml4j.nn.axons.PoolingAxonsConfig;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
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
	private DirectedComponentsContext directedComponentsContext;

	public DefaultDirectedComponentFactoryImpl(MatrixFactory matrixFactory, AxonsFactory axonsFactory,
			DifferentiableActivationFunctionFactory activationFunctionFactory, DirectedComponentsContext directedComponentsContext) {
		this.matrixFactory = matrixFactory;
		this.axonsFactory = axonsFactory;
		this.directedComponentFactory = this;
		this.activationFunctionFactory = activationFunctionFactory;
		this.directedComponentsContext = directedComponentsContext;
	}

	public void setDirectedComponentFactory(DirectedComponentFactory directedComponentFactory) {
		this.directedComponentFactory = directedComponentFactory;
	}

	@Override
	public DirectedAxonsComponent<Neurons, Neurons, ?> createFullyConnectedAxonsComponent(String name, FullyConnectedAxonsConfig axonsConfig,  WeightsMatrix connectionWeights, BiasVector biases) {
		return createTypedDirectedAxonsComponent(name,
				axonsFactory.createFullyConnectedAxons(axonsConfig.getAxonsConfig(),
						connectionWeights, 
						biases), axonsConfig.getAxonsContextConfigurer());
	}
	
	private <L extends Neurons, R extends Neurons, A extends Axons<L, R, A>> DirectedAxonsComponent<L, R, ?> createTypedDirectedAxonsComponent(String name, 
			A axons, AxonsContextConfigurer axonsContextConfiguer) {
		DirectedAxonsComponent<L, R, ?> directedAxonsComponent = new DefaultDirectedAxonsComponentImpl<>(name, axons);
		
		if (axonsContextConfiguer != null) {
			axonsContextConfiguer.accept(directedAxonsComponent.getContext(directedComponentsContext));
		}
		
		return directedAxonsComponent;
	}

	@Override
	public <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R, ?> createDirectedAxonsComponent(String name, 
			Axons<L, R, ?> axons, AxonsContextConfigurer axonsContextConfiguer) {
		if (axons instanceof TrainableAxons) {
			return createTypedDirectedAxonsComponent(name, new GenericTrainableAxonsAdapter<L, R>((TrainableAxons<L, R, ?>)axons), axonsContextConfiguer);
		} else {
			return createTypedDirectedAxonsComponent(name, new GenericAxonsAdapter<L, R>(axons), axonsContextConfiguer);
		}
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createConvolutionalAxonsComponent(String name, ConvolutionalAxonsConfig config, WeightsMatrix connectionWeights, BiasVector biases) {
		return createTypedDirectedAxonsComponent(name,
				axonsFactory.createConvolutionalAxons(config.getAxonsConfig(), connectionWeights, biases), config.getAxonsContextConfigurer());
		
		
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createMaxPoolingAxonsComponent(String name, PoolingAxonsConfig config, boolean scaleOutputs) {
		return createTypedDirectedAxonsComponent(name,
				axonsFactory.createMaxPoolingAxons(config.getAxonsConfig(), scaleOutputs), AxonsContextConfigurer.defaultConfigurer());
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createAveragePoolingAxonsComponent(String name, PoolingAxonsConfig config) {
		return createTypedDirectedAxonsComponent(name, axonsFactory.createAveragePoolingAxons(config.getAxonsConfig()), AxonsContextConfigurer.defaultConfigurer());
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, ?> createBatchNormAxonsComponent(String name, BatchNormAxonsConfig<N> config) {
		
		BatchNormDirectedAxonsComponent<N, ?>  component =  new DefaultBatchNormDirectedAxonsComponentImpl<>(name, 
				axonsFactory.createScaleAndShiftAxons(config.getAxonsConfig(),
						new WeightsMatrixImpl(expandChannelValuesToFeatureValues(matrixFactory, config.getNeurons(), config.getBatchNormConfig().getGammaColumnVector()),
								new WeightsFormatImpl(Arrays.asList(
										Dimension.INPUT_DEPTH, 
										Dimension.INPUT_HEIGHT, 
										Dimension.INPUT_WIDTH), 
										Arrays.asList(Dimension.OUTPUT_FEATURE),
										WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)),
						config.getBatchNormConfig().getBetaColumnVector() == null ? null : new BiasVectorImpl(expandChannelValuesToFeatureValues(matrixFactory, config.getNeurons(), config.getBatchNormConfig().getBetaColumnVector()), FeaturesVectorFormat.DEFAULT_BIAS_FORMAT)),
				expandChannelValuesToFeatureValues(matrixFactory, config.getNeurons(), config.getBatchNormConfig().getMeanColumnVector()),
				expandChannelValuesToFeatureValues(matrixFactory, config.getNeurons(), config.getBatchNormConfig().getVarianceColumnVector()));
		
		if (config.getBatchNormAxonsContextConfigurer() != null) {
			config.getBatchNormAxonsContextConfigurer() .accept(component.getContext(directedComponentsContext));
		}
		
		return component;
	}

	
	public static Matrix expandChannelValuesToFeatureValues(MatrixFactory matrixFactory, Neurons rightNeurons,
			WeightsMatrix channelValues) {
		if (channelValues == null) {
			return null;
		}
		if (!Dimension.isEquivalent(channelValues.getFormat().getInputDimensions(), Arrays.asList(Dimension.INPUT_DEPTH), DimensionScope.INPUT)) {
			throw new IllegalArgumentException("Expected batch norm to be of format with input dimensions:" +  Dimension.INPUT_DEPTH);
		}
		return expandChannelValuesToFeatureValues(matrixFactory, rightNeurons, channelValues.getMatrix());
	}
	
	
	
	public static Matrix expandChannelValuesToFeatureValues(MatrixFactory matrixFactory, Neurons rightNeurons,
			FeaturesVector channelValues) {
		if (channelValues == null) {
			return null;
		}
		if (!Dimension.isEquivalent(channelValues.getFormat().getDimensions(), Arrays.asList(Dimension.OUTPUT_DEPTH), DimensionScope.OUTPUT)) {
			throw new IllegalArgumentException("Expected batch norm to be of format with input dimensions:" +  Dimension.OUTPUT_DEPTH);
		}
		return expandChannelValuesToFeatureValues(matrixFactory, rightNeurons, channelValues.getVector());
	}

	public static Matrix expandChannelValuesToFeatureValues(MatrixFactory matrixFactory, Neurons rightNeurons,
			Matrix channelValues) {
		if (channelValues == null) {
			return null;
		}
		
		float[] channelValuesArray = channelValues.getRowByRowArray();

		
		if (rightNeurons instanceof Neurons3D) {
	
			Neurons3D neurons3D = (Neurons3D)rightNeurons;
			
		float[] channelValuesExpanded = new float[rightNeurons.getNeuronCountExcludingBias()];
		int index = 0;
		for (int channel = 0; channel < channelValuesArray.length; channel++) {
			for (int i = 0; i < neurons3D.getWidth() * neurons3D.getHeight(); i++) {
				channelValuesExpanded[index++] = channelValuesArray[channel];
			}
		}
		return matrixFactory.createMatrixFromRowsByRowsArray(channelValuesExpanded.length, 1, channelValuesExpanded);
		
		} else {
			return matrixFactory.createMatrixFromRowsByRowsArray(channelValuesArray.length, 1, channelValuesArray);

		}
	}

	@Override
	public <N extends Neurons> DirectedAxonsComponent<N, N, ?> createPassThroughAxonsComponent(String name, N leftNeurons,
			N rightNeurons) {
		return createDirectedAxonsComponent(name, new PassThroughAxonsImpl<>(leftNeurons, rightNeurons), AxonsContextConfigurer.defaultConfigurer());
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

	public DefaultDirectedComponentBatch createDirectedComponentBatch(
			List<DefaultChainableDirectedComponent<?, ?>> parallelComponents) {
		return new DefaultComponentBatchImpl(parallelComponents);
	}

	@Override
	public DefaultChainableDirectedComponent<?, ?> createComponent(String name, 
			Neurons leftNeurons, Neurons rightNeurons, NeuralComponentType neuralComponentType) {
		if (DefaultSpaceToDepthAxons.SPACE_TO_DEPTH_AXONS_TYPE.getId().equals(neuralComponentType.getId()) && leftNeurons instanceof Neurons3D && rightNeurons instanceof Neurons3D) {
			return createDirectedAxonsComponent(name, axonsFactory
					.createAxons3D(AxonsType.createCustomBaseType("SPACE_TO_DEPTH"), 
							DefaultSpaceToDepthAxons.class, new AxonsConfig<Neurons3D, Neurons3D>((Neurons3D)leftNeurons,(Neurons3D)rightNeurons)), AxonsContextConfigurer.defaultConfigurer());			
		}
		throw new UnsupportedOperationException("Creation of component by component type not yet implemented");
	}
}
