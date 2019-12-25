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
package org.ml4j.nn.components.onetoone.base;

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/**
 * Abstract base class for unit tests for implementations of ManyToOneDirectedComponent.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class DefaultDirectedComponentChainTestBase {

	@Mock
	private NeuronsActivation mockNeuronsActivation1;
	
	@Mock
	private MatrixFactory mockMatrixFactory;
	
	@Mock
	private NeuronsActivation mockNeuronsActivation2;
	
	@Mock
	private NeuronsActivation mockNeuronsActivation3;
	
	@Mock
	private DirectedComponentsContext mockDirectedComponentsContext;
	
	@Mock
	private DefaultChainableDirectedComponentActivation mockComponent1Activation;
	
	@Mock
	private DefaultChainableDirectedComponentActivation mockComponent2Activation;
	
	@Mock
	private DefaultChainableDirectedComponent<DefaultChainableDirectedComponentActivation, DirectedComponentsContext> mockComponent1;

	@Mock
	private DefaultChainableDirectedComponent<DefaultChainableDirectedComponentActivation, DirectedComponentsContext> mockComponent2;
	
	@Before
	public void setup() {
	    MockitoAnnotations.initMocks(this);
	    
	    Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(mockMatrixFactory);
	    
	    Mockito.when(mockNeuronsActivation1.getFeatureCount()).thenReturn(100);
	    Mockito.when(mockNeuronsActivation1.getExampleCount()).thenReturn(32);
	    Mockito.when(mockNeuronsActivation1.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	    Mockito.when(mockNeuronsActivation2.getFeatureCount()).thenReturn(200);
	    Mockito.when(mockNeuronsActivation2.getExampleCount()).thenReturn(32);
	    Mockito.when(mockNeuronsActivation2.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	    Mockito.when(mockNeuronsActivation3.getFeatureCount()).thenReturn(300);
	    Mockito.when(mockNeuronsActivation3.getExampleCount()).thenReturn(32);
	    Mockito.when(mockNeuronsActivation3.getFeatureOrientation()).thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	    
	    Mockito.when(mockComponent1.getInputNeurons()).thenReturn(new Neurons(100, false));
	    Mockito.when(mockComponent2.getInputNeurons()).thenReturn(new Neurons(200, false));
	    Mockito.when(mockComponent1.getOutputNeurons()).thenReturn(new Neurons(200, false));
	    Mockito.when(mockComponent2.getOutputNeurons()).thenReturn(new Neurons(300, false));
	   
	    
		Mockito.when(mockComponent1.forwardPropagate(Mockito.eq(mockNeuronsActivation1), Mockito.any())).thenReturn(mockComponent1Activation);
		Mockito.when(mockComponent2.forwardPropagate(Mockito.eq(mockNeuronsActivation2), Mockito.any())).thenReturn(mockComponent2Activation);
		Mockito.when(mockComponent1Activation.getOutput()).thenReturn(mockNeuronsActivation2);
		Mockito.when(mockComponent2Activation.getOutput()).thenReturn(mockNeuronsActivation3);

	}

	private DefaultDirectedComponentChain createDefaultDirectedComponentChain(List<DefaultChainableDirectedComponent<?, ?>> components) {
		return createDefaultDirectedComponentChainUnderTest(components);
	}
		
	protected abstract DefaultDirectedComponentChain createDefaultDirectedComponentChainUnderTest(List<DefaultChainableDirectedComponent<?, ?>> components);

	@Test
	public void testConstruction() {	
		
		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChain componentChain = createDefaultDirectedComponentChain(mockComponents);
		Assert.assertNotNull(componentChain);
	}
	
	@Test
	public void testGetComponentType() {	
		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChain componentChain = createDefaultDirectedComponentChain(mockComponents);
		Assert.assertNotNull(componentChain);
		Assert.assertEquals(DirectedComponentType.COMPONENT_CHAIN, componentChain.getComponentType());
	}
	
	@Test
	public void testForwardPropagate() {	
				
		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChain componentChain = createDefaultDirectedComponentChain(mockComponents);
		Assert.assertNotNull(componentChain);
		
		DefaultDirectedComponentChainActivation chainActivation =  componentChain.forwardPropagate(mockNeuronsActivation1, mockDirectedComponentsContext);
		Assert.assertNotNull(chainActivation);
		NeuronsActivation output = chainActivation.getOutput();
		Assert.assertNotNull(output);
		Assert.assertEquals(mockNeuronsActivation3.getExampleCount(), output.getExampleCount());
		Assert.assertEquals(mockNeuronsActivation3.getFeatureOrientation(), output.getFeatureOrientation());
	}
	
	@Test
	public void testDup() {
				
		// TODO

	}
	

}
