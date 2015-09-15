using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Trabalho_Pratico_I
{
    /*
     * Disciplina: Algoritmos em grafos
     * Professor: Rodrigo Caetano Rocha
     * Topico: Estruturas de Dados de Digrafos.
     */

    public abstract class BuscaLargura : Busca
    {
        private Digrafo g;
        private int vertice_Origem;

        public BuscaLargura(Digrafo g, int verticeOrigem) : base(g,verticeOrigem)
        {
        }

        protected override void realizaBusca()
        {
            //FAZER
        }

        public override Boolean existeCaminhoPara(int verticeDestino)
        {
            //FAZER
            return false;
        }

        public override List<int> caminhoPara(int verticeDestino)
        {
            //FAZER
            return null;
        }

        public override double pesoCaminhoPara(int verticeDestino)
        {
            //FAZER
            return double.NaN;
        }
    }
}
