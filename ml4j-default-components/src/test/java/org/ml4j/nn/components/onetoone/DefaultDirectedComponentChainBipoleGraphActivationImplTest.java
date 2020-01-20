package org.ml4j.nn.components.onetoone;

import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatchActivation;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraphActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBipoleGraphActivationTestBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;

public class DefaultDirectedComponentChainBipoleGraphActivationImplTest
		extends DefaultDirectedComponentChainBipoleGraphActivationTestBase {

	@Mock
	protected OneToManyDirectedComponentActivation mockInputLinkActivation;

	@Mock
	protected ManyToOneDirectedComponentActivation mockOutputLinkActivation;

	@Mock
	protected DefaultDirectedComponentChainBatchActivation mockEdgesActivation;

	@Override
	protected DefaultDirectedComponentChainBipoleGraphActivation createDefaultDirectedComponentChainBipoleGraphActivationUnderTest(
			DefaultDirectedComponentChainBipoleGraph bipoleGraph, NeuronsActivation output) {

		Mockito.when(mockOutputLinkActivation.getOutput()).thenReturn(output);

		return new DefaultDirectedComponentChainBipoleGraphActivationImpl(bipoleGraph, mockInputLinkActivation,
				mockEdgesActivation, mockOutputLinkActivation, false);
	}

	@Override
	public void testBackPropagate() {
		DirectedComponentGradient<List<NeuronsActivation>> batchGradient1 = MockTestData.mockBatchComponentGradient(110,
				32, 2);
		DirectedComponentGradient<List<NeuronsActivation>> batchGradient2 = MockTestData.mockBatchComponentGradient(110,
				32, 2);
		DirectedComponentGradient<NeuronsActivation> outputGradient = MockTestData.mockComponentGradient(110, 32, this);

		Mockito.when(mockOutputLinkActivation.backPropagate(mockInboundGradient)).thenReturn(batchGradient1);
		Mockito.when(mockEdgesActivation.backPropagate(batchGradient1)).thenReturn(batchGradient2);
		Mockito.when(mockInputLinkActivation.backPropagate(batchGradient2)).thenReturn(outputGradient);
		NeuronsActivation mockOutputGradientActivation = MockTestData.mockNeuronsActivation(100, 32);
		Mockito.when(outputGradient.getOutput()).thenReturn(mockOutputGradientActivation);
		super.testBackPropagate();
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
