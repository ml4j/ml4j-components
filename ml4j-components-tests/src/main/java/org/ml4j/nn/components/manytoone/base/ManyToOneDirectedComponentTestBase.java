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
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/**
 * Abstract base class for unit tests for implementations of ManyToOneDirectedComponent.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class ManyToOneDirectedComponentTestBase {
	
	protected NeuronsActivation neuronsActivation1;
	
	protected NeuronsActivation neuronsActivation2;
	
	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;
	
	@Before
	public void setup() {
	    MockitoAnnotations.initMocks(this);
	  
	    neuronsActivation1 = createNeuronsActivation1(100, 32);
	    neuronsActivation2 = createNeuronsActivation2(200, 32);
	    
	}
	
	protected abstract NeuronsActivation createNeuronsActivation1(int featureCount, int examples);
	protected abstract NeuronsActivation createNeuronsActivation2(int featureCount, int examples);


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
		Assert.assertEquals(DirectedComponentType.MANY_TO_ONE, manyToOneDirectedComponent.getComponentType());
	}
	
	@Test
	public void testForwardPropagate() {	
				
		ManyToOneDirectedComponent<?> manyToOneDirectedComponent = createManyToOneDirectedAxonsComponent(PathCombinationStrategy.FILTER_CONCAT);
		Assert.assertNotNull(manyToOneDirectedComponent);
		List<NeuronsActivation> mockNeuronActivations = Arrays.asList(neuronsActivation1, neuronsActivation2);
		ManyToOneDirectedComponentActivation activation =  manyToOneDirectedComponent.forwardPropagate(mockNeuronActivations, mockDirectedComponentsContext);
		Assert.assertNotNull(activation);
		NeuronsActivation output = activation.getOutput();
		Assert.assertNotNull(output);
		Assert.assertEquals(mockNeuronActivations.stream().mapToInt(a -> a.getFeatureCount()).sum(), output.getFeatureCount());
		Assert.assertEquals(neuronsActivation1.getExampleCount(), output.getExampleCount());
		Assert.assertEquals(neuronsActivation2.getExampleCount(), output.getExampleCount());
		Assert.assertEquals(neuronsActivation1.getFeatureOrientation(), output.getFeatureOrientation());
		Assert.assertEquals(neuronsActivation2.getFeatureOrientation(), output.getFeatureOrientation());
	}
	
	@Test
	public void testDup() {
				
		ManyToOneDirectedComponent<?> oneToManyDirectedComponent = createManyToOneDirectedAxonsComponent(PathCombinationStrategy.FILTER_CONCAT);

		ManyToOneDirectedComponent<?> dupComponent = oneToManyDirectedComponent.dup();
		
		Assert.assertNotNull(dupComponent);
		Assert.assertEquals(DirectedComponentType.MANY_TO_ONE, oneToManyDirectedComponent.getComponentType());
		
		List<NeuronsActivation> mockNeuronActivations = Arrays.asList(neuronsActivation1, neuronsActivation2);
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
