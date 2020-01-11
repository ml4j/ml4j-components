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
package org.ml4j.nn.components.manytoone.base;

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/**
 * Abstract base class for unit tests for implementations of ManyToOneDirectedComponent.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class ManyToOneDirectedComponentTestBase extends TestBase {
	
	
	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;
	
	@Before
	public void setup() {
	    MockitoAnnotations.initMocks(this);	    
	}

	private ManyToOneDirectedComponent<?> createManyToOneDirectedAxonsComponent(PathCombinationStrategy pathCombinationStrategy) {
		return createManyToOneDirectedComponentUnderTest(pathCombinationStrategy);
	}
		
	protected abstract ManyToOneDirectedComponent<?> createManyToOneDirectedComponentUnderTest(PathCombinationStrategy pathCombinationStrategy);

	@Test
	public void testConstruction() {	
		ManyToOneDirectedComponent<?> manyToOneDirectedComponent = createManyToOneDirectedAxonsComponent(PathCombinationStrategy.FILTER_CONCAT);
		Assert.assertNotNull(manyToOneDirectedComponent);
	}
	
	@Test
	public void testGetComponentType() {	
		ManyToOneDirectedComponent<?> manyToOneDirectedComponent = createManyToOneDirectedAxonsComponent(PathCombinationStrategy.FILTER_CONCAT);
		Assert.assertNotNull(manyToOneDirectedComponent);
		Assert.assertEquals(NeuralComponentType.MANY_TO_ONE, manyToOneDirectedComponent.getComponentType());
	}
		
	@Test
	public void testForwardPropagate() {	
		
		NeuronsActivation inputActivation1 = createNeuronsActivation(100, 32);
		NeuronsActivation inputActivation2 = createNeuronsActivation(100, 32);

		ManyToOneDirectedComponent<?> manyToOneDirectedComponent = createManyToOneDirectedAxonsComponent(PathCombinationStrategy.FILTER_CONCAT);
		Assert.assertNotNull(manyToOneDirectedComponent);
		
		List<NeuronsActivation> mockNeuronActivations = Arrays.asList(inputActivation1, inputActivation2);
		ManyToOneDirectedComponentActivation activation = manyToOneDirectedComponent.forwardPropagate(mockNeuronActivations, mockDirectedComponentsContext);
		Assert.assertNotNull(activation);
		
		NeuronsActivation output = activation.getOutput();
		
		Assert.assertNotNull(output);
		Assert.assertEquals(mockNeuronActivations.stream().mapToInt(a -> a.getFeatureCount()).sum(), output.getFeatureCount());
		Assert.assertEquals(inputActivation1.getExampleCount(), output.getExampleCount());
		Assert.assertEquals(inputActivation2.getExampleCount(), output.getExampleCount());
		Assert.assertEquals(inputActivation1.getFeatureOrientation(), output.getFeatureOrientation());
		Assert.assertEquals(inputActivation2.getFeatureOrientation(), output.getFeatureOrientation());
	}
	
	@Test
	public void testDup() {
				
		ManyToOneDirectedComponent<?> oneToManyDirectedComponent = createManyToOneDirectedAxonsComponent(PathCombinationStrategy.FILTER_CONCAT);

		ManyToOneDirectedComponent<?> dupComponent = oneToManyDirectedComponent.dup();
		
		Assert.assertNotNull(dupComponent);
		Assert.assertEquals(NeuralComponentType.MANY_TO_ONE, oneToManyDirectedComponent.getComponentType());
		
		NeuronsActivation inputActivation1 = MockTestData.mockNeuronsActivation(100, 32);
		NeuronsActivation inputActivation2 = MockTestData.mockNeuronsActivation(100, 32);
		
		List<NeuronsActivation> mockNeuronActivations = Arrays.asList(inputActivation1, inputActivation2);
		ManyToOneDirectedComponentActivation activation =  oneToManyDirectedComponent.forwardPropagate(mockNeuronActivations, mockDirectedComponentsContext);

		ManyToOneDirectedComponentActivation dupActivation =  dupComponent.forwardPropagate(mockNeuronActivations, mockDirectedComponentsContext);
		
		NeuronsActivation output = activation.getOutput();
		NeuronsActivation dupOutput = dupActivation.getOutput();

		Assert.assertNotNull(output);
		Assert.assertNotNull(dupOutput);

		Assert.assertEquals(output.getFeatureCount(), dupOutput.getFeatureCount());
		Assert.assertEquals(output.getExampleCount(), dupOutput.getExampleCount());

	}
	

}
