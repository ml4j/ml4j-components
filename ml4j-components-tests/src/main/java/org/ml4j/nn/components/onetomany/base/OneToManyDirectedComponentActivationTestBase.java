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
package org.ml4j.nn.components.onetomany.base;

import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class OneToManyDirectedComponentActivationTestBase extends TestBase {
	
	
	@Mock
	protected AxonsContext mockAxonsContext;
	
	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;

	@Mock
	protected OneToManyDirectedComponent<?> mockOneToManyDirectedComponent;
	
	
	@Before
	public void setup() {
	    MockitoAnnotations.initMocks(this);
	    Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);

	}

	private OneToManyDirectedComponentActivation createOneToManyDirectedComponentActivation(MatrixFactory matrixFactory, NeuronsActivation input, int outputCount) {
		return createOneToManyDirectedComponentActivationUnderTest(matrixFactory, input, outputCount);
	}
		
	protected abstract OneToManyDirectedComponentActivation createOneToManyDirectedComponentActivationUnderTest(MatrixFactory matrixFactory, NeuronsActivation input, int outputCount);
	@Test
	public void testConstruction() {
		
		NeuronsActivation mockInputActivation = MockTestData.mockNeuronsActivation(100, 32);
		
		OneToManyDirectedComponentActivation oneToManyDirectedComponentActivation = createOneToManyDirectedComponentActivation(matrixFactory, mockInputActivation, 2);
		Assert.assertNotNull(oneToManyDirectedComponentActivation);
	}
	
	@Test
	public void testGetOutput() {
		
		NeuronsActivation mockInputActivation = MockTestData.mockNeuronsActivation(100, 32);
				
		OneToManyDirectedComponentActivation oneToManyDirectedComponentActivation = createOneToManyDirectedComponentActivation(matrixFactory, mockInputActivation, 2);
		Assert.assertNotNull(oneToManyDirectedComponentActivation);
		Assert.assertNotNull(oneToManyDirectedComponentActivation.getOutput());
		Assert.assertEquals(2, oneToManyDirectedComponentActivation.getOutput().size());

		Assert.assertSame(mockInputActivation, oneToManyDirectedComponentActivation.getOutput().get(0));
		Assert.assertSame(mockInputActivation, oneToManyDirectedComponentActivation.getOutput().get(1));

	}
	
	@Test
	public void testBackPropagate() {
		
		NeuronsActivation mockInputActivation = MockTestData.mockNeuronsActivation(100, 32, matrixFactory);
		
		DirectedComponentGradient<List<NeuronsActivation>> mockInboundGradient = MockTestData.mockBatchComponentGradient(100, 32, 2, matrixFactory);
		
		OneToManyDirectedComponentActivation oneToManyDirectedComponentActivation = createOneToManyDirectedComponentActivation(matrixFactory, mockInputActivation, 2);
		
		Assert.assertNotNull(oneToManyDirectedComponentActivation);
		
		DirectedComponentGradient<NeuronsActivation> backPropagatedGradient = oneToManyDirectedComponentActivation.backPropagate(mockInboundGradient);
		
		Assert.assertNotNull(backPropagatedGradient);
		Assert.assertNotNull(backPropagatedGradient.getOutput());

		Assert.assertFalse(backPropagatedGradient.getOutput().getFeatureCount() == 0);
		Assert.assertFalse(backPropagatedGradient.getOutput().getExampleCount() == 0);

		Assert.assertEquals(100, backPropagatedGradient.getOutput().getFeatureCount());
		Assert.assertEquals(32, backPropagatedGradient.getOutput().getExampleCount());
	
	}


}
