import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const EnergyApp());
}

class EnergyApp extends StatelessWidget {
  const EnergyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IoT Energy Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  // ----- API endpoint (swap in your own Render URL) -----
  static const String apiUrl =
      'https://ghana-pm25-api.onrender.com/predict';

  // Device power modes
  static const Map<int, String> actions = {
    0: 'SLEEP_DEEP',
    1: 'SLEEP_LIGHT',
    2: 'ACTIVE_LOW',
    3: 'TX_LOW',
    4: 'TX_MED',
    5: 'TX_HIGH',
  };

  int _selectedAction = 5;
  final _cpu = TextEditingController(text: '0.65');
  final _mem = TextEditingController(text: '0.40');
  final _signal = TextEditingController(text: '0.75');
  final _queue = TextEditingController(text: '12');
  final _temp = TextEditingController(text: '9.5');

  bool _loading = false;
  String? _resultText;
  String? _note;
  String? _errorText;
  Color _resultColor = Colors.indigo;

  @override
  void dispose() {
    _cpu.dispose();
    _mem.dispose();
    _signal.dispose();
    _queue.dispose();
    _temp.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    setState(() {
      _loading = true;
      _resultText = null;
      _note = null;
      _errorText = null;
    });

    final cpu = double.tryParse(_cpu.text.trim());
    final mem = double.tryParse(_mem.text.trim());
    final signal = double.tryParse(_signal.text.trim());
    final queue = int.tryParse(_queue.text.trim());
    final temp = double.tryParse(_temp.text.trim());

    // Client-side validation before hitting the API
    if (cpu == null || mem == null || signal == null || queue == null || temp == null) {
      setState(() {
        _loading = false;
        _errorText = 'Please enter valid numbers in every field.';
      });
      return;
    }
    if (cpu < 0 || cpu > 1 || mem < 0 || mem > 1 || signal < 0 || signal > 1) {
      setState(() {
        _loading = false;
        _errorText = 'CPU, memory, and signal must be between 0 and 1.';
      });
      return;
    }
    if (queue < 0 || queue > 100) {
      setState(() {
        _loading = false;
        _errorText = 'Queue size must be between 0 and 100.';
      });
      return;
    }
    if (temp < -20 || temp > 60) {
      setState(() {
        _loading = false;
        _errorText = 'Temperature must be between -20 and 60 °C.';
      });
      return;
    }

    try {
      final response = await http
          .post(
            Uri.parse(apiUrl),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'cpu_usage': cpu,
              'memory_usage': mem,
              'signal_quality': signal,
              'action': _selectedAction,
              'queue_size': queue,
              'temperature_C': temp,
            }),
          )
          .timeout(const Duration(seconds: 90));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _loading = false;
          _resultText = '${data['predicted_energy_mJ']} mJ';
          _note = data['efficiency_note'] ?? '';
          _resultColor = _colorFor(data['predicted_energy_mJ']);
        });
      } else if (response.statusCode == 422) {
        final data = jsonDecode(response.body);
        String message;
        final detail = data['detail'];
        if (detail is String) {
          message = detail;
        } else if (detail is List && detail.isNotEmpty) {
          message = detail[0]['msg']?.toString() ?? 'Invalid input.';
        } else {
          message = 'Invalid input. Please check your values.';
        }
        setState(() {
          _loading = false;
          _errorText = message;
        });
      } else {
        setState(() {
          _loading = false;
          _errorText = 'Server error (${response.statusCode}). Please try again.';
        });
      }
    } catch (e) {
      setState(() {
        _loading = false;
        _errorText =
            'Could not reach the server. It may be waking up — wait a moment and try again.';
      });
    }
  }

  Color _colorFor(dynamic mj) {
    final v = (mj is num) ? mj.toDouble() : 0.0;
    if (v <= 100) return Colors.green;
    if (v <= 400) return Colors.lightGreen;
    if (v <= 800) return Colors.orange;
    return Colors.red;
  }

  Widget _numField(String label, TextEditingController c, String hint) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          TextField(
            controller: c,
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            decoration: InputDecoration(
              border: const OutlineInputBorder(),
              hintText: hint,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('IoT Edge-Device Energy Predictor'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const SizedBox(height: 8),
            const Text(
              'Predict the energy an IoT node will consume (mJ) from its operating state.',
              style: TextStyle(fontSize: 15, color: Colors.black54),
            ),
            const SizedBox(height: 24),

            // Action dropdown
            const Text('Power mode (action)',
                style: TextStyle(fontWeight: FontWeight.w600)),
            const SizedBox(height: 6),
            DropdownButtonFormField<int>(
              value: _selectedAction,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                contentPadding:
                    EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              ),
              items: actions.entries
                  .map((e) => DropdownMenuItem(
                        value: e.key,
                        child: Text('${e.key} — ${e.value}'),
                      ))
                  .toList(),
              onChanged: (v) {
                if (v != null) setState(() => _selectedAction = v);
              },
            ),
            const SizedBox(height: 16),

            _numField('CPU usage', _cpu, '0.0 – 1.0'),
            _numField('Memory usage', _mem, '0.0 – 1.0'),
            _numField('Signal quality', _signal, '0.0 – 1.0'),
            _numField('Queue size', _queue, '0 – 100'),
            _numField('Temperature (°C)', _temp, '-20 – 60'),

            const SizedBox(height: 8),
            SizedBox(
              height: 50,
              child: ElevatedButton(
                onPressed: _loading ? null : _predict,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.indigo,
                  foregroundColor: Colors.white,
                ),
                child: _loading
                    ? const SizedBox(
                        height: 22,
                        width: 22,
                        child: CircularProgressIndicator(
                            strokeWidth: 2.5, color: Colors.white),
                      )
                    : const Text('Predict', style: TextStyle(fontSize: 17)),
              ),
            ),
            const SizedBox(height: 28),

            if (_resultText != null)
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: _resultColor.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: _resultColor, width: 1.5),
                ),
                child: Column(
                  children: [
                    const Text('Predicted energy consumed',
                        style: TextStyle(fontSize: 15, color: Colors.black54)),
                    const SizedBox(height: 8),
                    Text(
                      _resultText!,
                      style: TextStyle(
                        fontSize: 34,
                        fontWeight: FontWeight.bold,
                        color: _resultColor,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      _note ?? '',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        color: _resultColor,
                      ),
                    ),
                  ],
                ),
              ),

            if (_errorText != null)
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.red.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.red.shade300),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.error_outline, color: Colors.red),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        _errorText!,
                        style: const TextStyle(color: Colors.red, fontSize: 15),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}