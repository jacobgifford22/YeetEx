using Microsoft.AspNetCore.Mvc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Intex_2.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Intex_2.Controllers
{
    public class InferenceController : Controller
    {
        private InferenceSession _session;

        private InferenceSession _session2;

        public InferenceController()
        {
            _session = new InferenceSession("wwwroot/model_severity.onnx");
            _session2 = new InferenceSession("wwwroot/model_county_accidents.onnx"); 
        }

        [HttpGet]
        public IActionResult MLModel()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Score(CrashData data)
        {
            var result = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("float_input", data.AsTensor())
            });
            Tensor<string> score = result.First().AsTensor<string>();
            var prediction = new Prediction { PredictedValue = score.First() };
            result.Dispose();
            return View("Score", prediction);
        }

        [HttpGet]
        public IActionResult CrashCount()
        {
            return View(); 
        }

        [HttpPost]
        public IActionResult Crash_Score(CrashCount data)
        {
            var result = _session2.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("float_input", data.AsTensor())
            });
            Tensor<float> score = result.First().AsTensor<float>();
            var prediction = new Prediction { PredictedValue = ((int) score.First()).ToString() };
            result.Dispose();
            return View("Crash_Score", prediction);
        }
    }
}
