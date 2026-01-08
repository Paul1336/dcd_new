import subprocess
import sys


def run_commands_groups_parallel(commands_groups):
    try:
        running_cmds = []  # [(process, remaining_commands)]
        for i, commands in enumerate(commands_groups):
            print(f"Starting command set {i+1}")
            cmd = commands[0]
            print("-" * 100)
            print(f"Running: {cmd}")
            print("-" * 100)
            process = subprocess.Popen(cmd, shell=True, text=True)
            running_cmds.append((process, commands[1:]))

        while running_cmds:
            for item in running_cmds[:]:  # Copy list for safe removal
                process, remaining = item
                ret = process.poll()
                if ret is not None:  # Command finished
                    if ret != 0:
                        print(f"Command failed: {process.args}")
                        running_cmds.remove(item)
                        continue

                    if not remaining:  # No more commands
                        print(f"Command set completed successfully")
                        running_cmds.remove(item)
                    else:
                        # Start next command
                        cmd = remaining[0]
                        print("-" * 100)
                        print(f"Running: {cmd}")
                        print("-" * 100)
                        process = subprocess.Popen(cmd, shell=True, text=True)
                        running_cmds[running_cmds.index(item)] = (
                            process,
                            remaining[1:],
                        )

    except KeyboardInterrupt:
        print("\nStopping all processes...")
        count = 0
        for process, _ in running_cmds:
            process.terminate()
            count += 1
        print(f"Stopped {count} processes")
        sys.exit(1)


def verify_commands_groups(commands_groups):
    for i, commands_group in enumerate(commands_groups):
        print(f"Group {i + 1}: {len(commands_group)} commands")
        for j, command in enumerate(commands_group):
            print(command)
        print("-" * 100)

    print(f"Total command groups: {len(commands_groups)}")

    input("Press Enter to continue... (Ctrl+C to cancel)")


def write_args_to_file(args, run_name):
    # Write args to a file
    print("Run name: ", run_name)
    with open(f"runs/{run_name}/args.json", "w") as f:
        import json

        json.dump(vars(args), f)
