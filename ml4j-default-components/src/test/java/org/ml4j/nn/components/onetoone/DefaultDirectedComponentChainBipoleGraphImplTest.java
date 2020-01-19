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
package org.ml4j.nn.components.onetoone;

import java.util.Arrays;
import java.util.List;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatchActivation;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBipoleGraphBase;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBipoleGraphTestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons1D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;

/**
 * Unit test for DefaultDirectedComponentChainBipoleGraphImpl.
 * 
 * @author Michael Lavelle
 *
 */
public class DefaultDirectedComponentChainBipoleGraphImplTest extends DefaultDirectedComponentChainBipoleGraphTestBase {

	@Mock
	private DefaultDirectedComponentChainBatch mockComponentChainBatch;
	
	@Mock
	private DefaultDirectedComponentChainBatchActivation mockComponentChainBatchActivation;
	
	@SuppressWarnings("rawtypes")
	@Mock
	private OneToManyDirectedComponent mockOneToManyDirectedComponent;
	
	@SuppressWarnings("rawtypes")
	@Mock
	private ManyToOneDirectedComponent mockManyToOneDirectedComponent;
	
	@Mock
	private OneToManyDirectedComponentActivation mockOneToManyDirectedComponentActivation;
	
	@Mock
	private ManyToOneDirectedComponentActivation mockManyToOneDirectedComponentActivation;
	
	@SuppressWarnings("unchecked")
	@Before
	@Override
	public void setup() {
		super.setup();
		Mockito.when(mockDirectedComponentFactory.createOneToManyDirectedComponent(Mockito.any())).thenReturn(mockOneToManyDirectedComponent);
		Mockito.when(mockDirectedComponentFactory.createManyToOneDirectedComponent((Neurons1D)Mockito.any(), Mockito.any())).thenReturn(mockManyToOneDirectedComponent);
	}
	
	@SuppressWarnings("unchecked")
	@Override
	protected DefaultDirectedComponentChainBipoleGraphBase createDefaultDirectedComponentChainBipoleGraphUnderTest(
			DirectedComponentFactory factory, List<DefaultChainableDirectedComponent<?, ?>> components,
			PathCombinationStrategy pathCombinationStrategy) {
				
		Mockito.when(mockOneToManyDirectedComponent.forwardPropagate(mockInputActivation, mockDirectedComponentsContext)).thenReturn(mockOneToManyDirectedComponentActivation);
		Mockito.when(mockManyToOneDirectedComponent.forwardPropagate(Arrays.asList(mockNeuronsActivation3, mockNeuronsActivation4), mockDirectedComponentsContext)).thenReturn(mockManyToOneDirectedComponentActivation);
		
		Mockito.when(mockOneToManyDirectedComponentActivation.getOutput()).thenReturn(Arrays.asList(mockNeuronsActivation1, mockNeuronsActivation2));
		
		NeuronsActivation mockOutput = MockTestData.mockNeuronsActivation(100, 32);
		
		Mockito.when(mockManyToOneDirectedComponentActivation.getOutput()).thenReturn(mockOutput);
		
		Mockito.when(mockComponentChainBatch.forwardPropagate(Arrays.asList(mockNeuronsActivation1, mockNeuronsActivation2), mockDirectedComponentsContext))
		.thenReturn(mockComponentChainBatchActivation);
		Mockito.when(mockComponentChainBatchActivation.getOutput()).thenReturn(Arrays.asList(mockNeuronsActivation3, mockNeuronsActivation4));
		return new DefaultDirectedComponentChainBipoleGraphImpl(factory, new Neurons1D(10, false), new Neurons1D(100, false), mockComponentChainBatch, pathCombinationStrategy);
	}

	
	@Override
	protected MatrixFactory createMatrixFactory() {
		return Mockito.mock(MatrixFactory.class);
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return MockTestData.mockNeuronsActivation(featureCount, exampleCount);
	}

}
